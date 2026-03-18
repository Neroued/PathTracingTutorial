#pragma once

#include "pt.h"

#include <cuda_runtime.h>

#include "math/vecmath.h"
#include "math/math_utils.h"
#include "math/sampling.h"
#include "core/ray.h"
#include "core/interaction.h"
#include "geometry/vertex.h"
#include "accel/bvh_traversal.cuh"
#include "materials/material.h"
#include "materials/microfacet.cuh"
#include "materials/disney_bsdf.cuh"
#include "lights/envmap.h"
#include "lights/light_sampler.h"
#include "lights/punctual_light.h"
#include "lights/light_pdf.cuh"
#include "samplers/sampler.cuh"
#include "scene/device_scene.h"

namespace pt {

struct DirectLightSample {
    vec3  wi       = vec3(0.0f);
    vec3  radiance = vec3(0.0f);
    float pdf      = 0.0f;
    float distance = Infinity;
    bool  valid    = false;
};

// ---- Ray helpers ----

__device__ inline vec3 offsetRayOrigin(const vec3& point, const vec3& normal, const vec3& direction) {
    constexpr float RayEps = 1e-4f;
    float sign = dot(direction, normal) >= 0.0f ? 1.0f : -1.0f;
    return point + normal * (RayEps * sign);
}

__device__ inline bool isOccluded(const DeviceSceneView& scene,
                                  const vec3& point,
                                  const vec3& normal,
                                  const vec3& direction,
                                  float maxDistance) {
    Ray shadow(offsetRayOrigin(point, normal, direction), direction);
    constexpr float ShadowEps = 1e-3f;
    float tMax = (maxDistance < Infinity) ? (maxDistance - ShadowEps) : Infinity;
    return occludedScene(shadow, scene.bvhNodes, scene.vertices, scene.faces, tMax);
}

// ---- Alias table O(1) sampling ----

__device__ inline uint32_t sampleAlias(const AliasEntry* table, uint32_t n, float u1, float u2) {
    uint32_t idx = static_cast<uint32_t>(u1 * n);
    if (idx >= n) idx = n - 1;
    return (u2 < table[idx].prob) ? idx : table[idx].alias;
}

__device__ inline uint32_t sampleEmissiveTriangleIndex(const DeviceSceneView& scene,
                                                       Sampler& sampler) {
    return sampleAlias(scene.triangleAlias, scene.numEmissiveTriangles,
                       sampler.get1D(), sampler.get1D());
}

__device__ inline uint32_t sampleEnvTexelIndex(const DeviceSceneView& scene,
                                               Sampler& sampler) {
    return sampleAlias(scene.envAlias, scene.numEnvTexels,
                       sampler.get1D(), sampler.get1D());
}

// ---- Light sampling ----

__device__ inline DirectLightSample sampleAreaLight(const DeviceSceneView& scene,
                                                    const SurfaceInteraction& si,
                                                    const vec3& normal,
                                                    Sampler& sampler) {
    DirectLightSample sample;
    if (scene.numEmissiveTriangles == 0 || scene.emissiveTriangles == nullptr) return sample;

    const EmissiveTriangleRef& ref =
        scene.emissiveTriangles[sampleEmissiveTriangleIndex(scene, sampler)];
    const TriangleFace& face = scene.faces[ref.triangleIndex];
    const Vertex& va = scene.vertices[face.v0];
    const Vertex& vb = scene.vertices[face.v1];
    const Vertex& vc = scene.vertices[face.v2];

    vec3 lightPoint;
    vec3 lightNormal;
    vec2 triSample = sampler.get2D();
    sampleTriangleSurface(va, vb, vc, triSample.x, triSample.y, lightPoint, lightNormal);

    vec3 toLight  = lightPoint - si.point;
    float dist2   = dot(toLight, toLight);
    if (dist2 <= 1e-8f) return sample;

    float dist    = pt::sqrt(dist2);
    vec3 wi       = toLight / dist;
    float cosSurf = pt::max(0.0f, dot(normal, wi));
    if (cosSurf <= 0.0f) return sample;

    float area = faceArea(va, vb, vc);
    if (area <= 0.0f) return sample;

    float cosLight = pt::abs(dot(lightNormal, -wi));
    if (cosLight <= 1e-5f) return sample;

    float pdfSolidAngle = (ref.selectionPmf / area) * dist2 / cosLight;
    if (pdfSolidAngle <= 0.0f) return sample;

    if (isOccluded(scene, si.point, normal, wi, dist)) return sample;

    const Material& lightMat = scene.materials[face.materialID];
    sample.wi       = wi;
    sample.radiance = emittedRadiance(lightMat);
    sample.pdf      = pdfSolidAngle;
    sample.distance = dist;
    sample.valid    = true;
    return sample;
}

__device__ inline DirectLightSample sampleEnvironmentLight(const DeviceSceneView& scene,
                                                           const SurfaceInteraction& si,
                                                           const vec3& normal,
                                                           Sampler& sampler) {
    DirectLightSample sample;
    if (scene.hdrTex == 0 || scene.envAlias == nullptr || scene.numEnvTexels == 0 ||
        scene.envWidth <= 0 || scene.envHeight <= 0) {
        return sample;
    }

    uint32_t texel = sampleEnvTexelIndex(scene, sampler);
    int x = static_cast<int>(texel % static_cast<uint32_t>(scene.envWidth));
    int y = static_cast<int>(texel / static_cast<uint32_t>(scene.envWidth));

    vec2 uv((static_cast<float>(x) + sampler.get1D()) / static_cast<float>(scene.envWidth),
            (static_cast<float>(y) + sampler.get1D()) / static_cast<float>(scene.envHeight));
    vec3 wi = directionFromSphericalMap(uv);

    float cosSurf = pt::max(0.0f, dot(normal, wi));
    if (cosSurf <= 0.0f) return sample;

    float pdf = environmentPdf(scene, wi);
    if (pdf <= 0.0f) return sample;

    if (isOccluded(scene, si.point, normal, wi, Infinity)) return sample;

    sample.wi       = wi;
    sample.radiance = sampleEnvironmentRadiance(scene.hdrTex, wi);
    sample.pdf      = pdf;
    sample.valid    = true;
    return sample;
}

// ---- Punctual light contribution (delta distribution -- no MIS) ----

__device__ inline vec3 evaluatePunctualLights(const DeviceSceneView& scene,
                                               const SurfaceInteraction& si,
                                               const MaterialEval& meval,
                                               const vec3& wo,
                                               const vec3& normal,
                                               const vec3& tangent) {
    if (scene.numPunctualLights == 0 || scene.punctualLights == nullptr)
        return vec3(0.0f);

    vec3 Ld(0.0f);
    for (uint32_t i = 0; i < scene.numPunctualLights; ++i) {
        const PunctualLight& light = scene.punctualLights[i];

        vec3 wi;
        float dist = Infinity;
        vec3 intensity(0.0f);

        if (light.type == LightType::Point || light.type == LightType::Spot) {
            vec3 toLight = light.position - si.point;
            float dist2  = dot(toLight, toLight);
            if (dist2 < 1e-8f) continue;
            dist = pt::sqrt(dist2);
            wi   = toLight / dist;
            float atten = distanceAttenuation(dist, light.range);
            if (light.type == LightType::Spot)
                atten *= spotAttenuation(light, toLight);
            intensity = light.color * (light.intensity * atten);
        } else {
            wi   = -light.direction;
            dist = Infinity;
            intensity = light.color * light.intensity;
        }

        float NoL = dot(normal, wi);
        if (NoL <= 0.0f) continue;

        if (isOccluded(scene, si.point, normal, wi, dist)) continue;

        BsdfEval bsdf = evalDisneyBsdf(meval, wo, wi, normal, tangent);
        Ld += intensity * bsdf.f;
    }
    return Ld;
}

// ---- MIS-weighted direct light evaluation ----

__device__ inline vec3 evaluateDirectLight(const DirectLightSample& lightSample,
                                           const MaterialEval& meval,
                                           const vec3& wo,
                                           const vec3& normal,
                                           const vec3& tangent,
                                           float combinedPdfScale) {
    if (!lightSample.valid) return vec3(0.0f);

    BsdfEval bsdf = evalDisneyBsdf(meval, wo, lightSample.wi, normal, tangent);
    if (bsdf.pdf <= 0.0f) return vec3(0.0f);

    float scaledPdf = lightSample.pdf * combinedPdfScale;
    float mis = powerHeuristic(scaledPdf, bsdf.pdf);
    return lightSample.radiance * bsdf.f * (mis / (scaledPdf + 1e-7f));
}

__device__ inline vec3 estimateDirectLighting(const DeviceSceneView& scene,
                                              const SurfaceInteraction& si,
                                              const MaterialEval& meval,
                                              const vec3& wo,
                                              const vec3& normal,
                                              const vec3& tangent,
                                              Sampler& sampler) {
    bool hasArea = (scene.numEmissiveTriangles > 0 && scene.emissiveTriangles != nullptr);
    bool hasEnv  = (scene.hdrTex != 0 && scene.envAlias != nullptr && scene.numEnvTexels > 0 &&
                    scene.envWidth > 0 && scene.envHeight > 0);

    if (!hasArea && !hasEnv) return vec3(0.0f);

    if (hasArea && hasEnv) {
        vec3 Ld(0.0f);
        Ld += evaluateDirectLight(sampleAreaLight(scene, si, normal, sampler),
                                  meval, wo, normal, tangent, 1.0f);
        Ld += evaluateDirectLight(sampleEnvironmentLight(scene, si, normal, sampler),
                                  meval, wo, normal, tangent, 1.0f);
        return Ld;
    }

    if (hasArea) {
        return evaluateDirectLight(sampleAreaLight(scene, si, normal, sampler),
                                   meval, wo, normal, tangent, 1.0f);
    }

    return evaluateDirectLight(sampleEnvironmentLight(scene, si, normal, sampler),
                               meval, wo, normal, tangent, 1.0f);
}

} // namespace pt
