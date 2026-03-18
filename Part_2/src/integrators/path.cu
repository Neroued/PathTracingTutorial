#include "pt.h"

#include <cuda_runtime.h>
#include <texture_indirect_functions.h>

#include "math/vecmath.h"
#include "math/math_utils.h"
#include "math/frame.h"
#include "math/sampling.h"
#include "core/ray.h"
#include "core/interaction.h"
#include "geometry/vertex.h"
#include "accel/bvh_traversal.cuh"
#include "materials/material.h"
#include "materials/disney_bsdf.cuh"
#include "materials/shading_utils.cuh"
#include "lights/envmap.h"
#include "lights/light_sampler.h"
#include "lights/light_pdf.cuh"
#include "camera/camera.h"
#include "samplers/sampler.cuh"
#include "integrators/direct_lighting.cuh"
#include "scene/device_scene.h"

namespace pt {

static __constant__ SceneParams d_params;

// ---- Camera ray generation ----
__device__ static Ray emitRay(Sampler& sampler) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    vec2 jitter = sampler.get2D();
    float u = (2.0f * (x + jitter.x) / d_params.width  - 1.0f)
              * d_params.camera.tanHalfFov * d_params.camera.aspect;
    float v = (2.0f * (y + jitter.y) / d_params.height - 1.0f)
              * d_params.camera.tanHalfFov;

    vec3 pos(d_params.camera.position.x, d_params.camera.position.y, d_params.camera.position.z);
    vec3 fwd(d_params.camera.forward.x,  d_params.camera.forward.y,  d_params.camera.forward.z);
    vec3 rgt(d_params.camera.right.x,    d_params.camera.right.y,    d_params.camera.right.z);
    vec3 cup(d_params.camera.up.x,       d_params.camera.up.y,       d_params.camera.up.z);

    vec3 dir = normalize(u * rgt + v * cup + fwd);

    if (d_params.camera.aperture > 0.0f) {
        vec3 focalPoint = pos + dir * (d_params.camera.focalDistance / dot(dir, fwd));
        vec2 lens = sampler.get2D();
        float r     = pt::sqrt(lens.x) * d_params.camera.aperture;
        float theta = TwoPi * lens.y;
        vec3 lensOffset = rgt * (r * pt::cos(theta)) + cup * (r * pt::sin(theta));
        pos = pos + lensOffset;
        dir = normalize(focalPoint - pos);
    }

    return Ray(pos, dir);
}

// ---- Path tracing core ----
__device__ static vec3 pathTrace(const DeviceSceneView& scene, Sampler& sampler) {
    vec3 Lo(0.0f, 0.0f, 0.0f);
    vec3 throughput(1.0f, 1.0f, 1.0f);
    Ray ray = emitRay(sampler);
    float lastBsdfPdf = 0.0f;
    bool lastWasDelta = true;

    for (int d = 0; d < d_params.maxDepth; ++d) {
        SurfaceInteraction si = intersectScene(ray, scene.bvhNodes,
                                               scene.vertices, scene.faces);

        if (!si.hit) {
            vec3 env = sampleEnvironmentRadiance(scene.hdrTex, ray.direction);
            if (env != vec3(0.0f)) {
                float misWeight = 1.0f;
                if (d > 0 && !lastWasDelta) {
                    float lightPdf = environmentPdf(scene, ray.direction);
                    misWeight = powerHeuristic(lastBsdfPdf, lightPdf);
                }
                Lo += throughput * env * misWeight;
            }
            break;
        }

        const Material& mat = scene.materials[si.materialId];
        bool entering       = dot(si.normal, ray.direction) < 0.0f;

        vec3 geomNormal;
        if (mat.doubleSided && !entering) {
            geomNormal = -si.normal;
            entering = true;
        } else {
            geomNormal = entering ? si.normal : -si.normal;
        }
        vec3 wo = -ray.direction;

        vec2 texUv = applyUvTransform(mat, si.uv);

        vec4 meshTangent = interpolateTangent(scene, si);
        if (!entering && mat.doubleSided)
            meshTangent = vec4(-meshTangent.x, -meshTangent.y, -meshTangent.z, -meshTangent.w);
        vec3 normal = applyNormalMap(mat, geomNormal, meshTangent, texUv, scene);

        MaterialEval meval = fetchMaterial(mat, texUv, scene.textures, scene.numTextures);

        if (isEmissive(mat)) {
            float misWeight = 1.0f;
            if (d > 0 && !lastWasDelta) {
                float lightPdf = emissiveTrianglePdf(scene, si.primitiveId, si.distance, ray.direction);
                misWeight = powerHeuristic(lastBsdfPdf, lightPdf);
            }
            Lo += throughput * emittedRadiance(mat) * misWeight;
            break;
        }

        vec3 tangent;
        {
            vec3 T(meshTangent.x, meshTangent.y, meshTangent.z);
            bool hasMeshT = (T.x != 0.0f || T.y != 0.0f || T.z != 0.0f);
            if (hasMeshT) {
                tangent = normalize(T - normal * dot(normal, T));
            } else {
                Frame fr(normal);
                tangent = fr.tangent;
            }
            if (meval.anisotropyRotation != 0.0f) {
                float cosA = pt::cos(meval.anisotropyRotation);
                float sinA = pt::sin(meval.anisotropyRotation);
                vec3 B = cross(normal, tangent);
                tangent = normalize(tangent * cosA + B * sinA);
            }
        }

        // Alpha handling
        if (mat.alphaMode == 1) { // MASK
            float alpha = 1.0f;
            if (mat.baseColorTexId >= 0 && scene.textures != nullptr &&
                static_cast<uint32_t>(mat.baseColorTexId) < scene.numTextures) {
                float4 tc = tex2D<float4>(scene.textures[mat.baseColorTexId], si.uv.x, si.uv.y);
                alpha = tc.w;
            }
            if (alpha < mat.alphaCutoff) {
                ray = Ray(si.point + ray.direction * 1e-4f, ray.direction);
                continue;
            }
        } else if (mat.alphaMode == 2) { // BLEND -- stochastic transparency
            float alpha = mat.baseAlpha;
            if (mat.baseColorTexId >= 0 && scene.textures != nullptr &&
                static_cast<uint32_t>(mat.baseColorTexId) < scene.numTextures) {
                float4 tc = tex2D<float4>(scene.textures[mat.baseColorTexId], si.uv.x, si.uv.y);
                alpha *= tc.w;
            }
            if (sampler.get1D() >= alpha) {
                ray = Ray(si.point + ray.direction * 1e-4f, ray.direction);
                continue;
            }
        }

        // Direct lighting (NEE) -- skip for delta-only surfaces
        bool isDeltaSurface = (meval.metallic >= 1.0f && meval.roughness < 0.01f) ||
                              (meval.specTrans >= 1.0f && meval.roughness < 0.01f);
        if (!isDeltaSurface) {
            Lo += throughput * estimateDirectLighting(scene, si, meval, wo, normal, tangent, sampler);
            Lo += throughput * evaluatePunctualLights(scene, si, meval, wo, normal, tangent);
        }

        // Sample BSDF
        BsdfSample bsdfSample = sampleDisneyBsdf(meval, wo, normal, tangent, sampler, entering);
        if (!bsdfSample.valid) break;

        throughput *= bsdfSample.weight;

        bool transmitted = dot(bsdfSample.wi, geomNormal) < 0.0f;
        vec3 offsetN = transmitted ? -geomNormal : geomNormal;
        ray = Ray(si.point + offsetN * 1e-4f, bsdfSample.wi);

        // Beer-Lambert volume attenuation (KHR_materials_volume)
        if (transmitted && meval.thicknessFactor > 0.0f && meval.attenuationDistance < 1e20f) {
            float t = meval.thicknessFactor;
            float ad = meval.attenuationDistance;
            vec3 ac = meval.attenuationColor;
            throughput *= vec3(
                pt::exp(-t * pt::max(-pt::log(pt::max(ac.x, 1e-6f)), 0.0f) / ad),
                pt::exp(-t * pt::max(-pt::log(pt::max(ac.y, 1e-6f)), 0.0f) / ad),
                pt::exp(-t * pt::max(-pt::log(pt::max(ac.z, 1e-6f)), 0.0f) / ad));
        }

        lastBsdfPdf  = bsdfSample.pdf;
        lastWasDelta = bsdfSample.delta;

        // Russian roulette
        if (d > 2) {
            float maxComp     = pt::max(pt::max(throughput.x, throughput.y), throughput.z);
            float surviveProb = pt::min(maxComp, 0.95f);
            if (sampler.get1D() >= surviveProb) break;
            throughput /= surviveProb;
        }
    }

    constexpr float clampVal = 100.0f;
    Lo.x = pt::min(Lo.x, clampVal);
    Lo.y = pt::min(Lo.y, clampVal);
    Lo.z = pt::min(Lo.z, clampVal);
    return Lo;
}

// ---- Kernel: path trace + in-kernel accumulation ----
__global__ static void kernelPathTrace(cudaSurfaceObject_t surfNew,
                                       cudaSurfaceObject_t surfAcc,
                                       DeviceSceneView scene) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= scene.width || y >= scene.height) return;

    vec3 acc(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < d_params.samplePerFrame; ++i) {
        Sampler sampler;
        if (d_params.samplerType == 1) {
            uint32_t globalIdx = scene.frameCount * static_cast<uint32_t>(d_params.samplePerFrame)
                               + static_cast<uint32_t>(i);
            sampler.initSobol(d_params.samplerType, x, y, globalIdx);
        } else {
            sampler.initPCG(d_params.samplerType, scene.frameCount, static_cast<unsigned int>(i));
        }
        acc += pathTrace(scene, sampler);
    }
    acc /= static_cast<float>(d_params.samplePerFrame);

    int byteOffset = x * static_cast<int>(sizeof(float4));
    float4 newSample = make_float4(acc, 1.0f);
    surf2Dwrite(newSample, surfNew, byteOffset, y);

    float4 accSample;
    surf2Dread(&accSample, surfAcc, byteOffset, y);
    float inv = 1.0f / (scene.frameCount + 1);
    accSample.x = accSample.x * (1.0f - inv) + newSample.x * inv;
    accSample.y = accSample.y * (1.0f - inv) + newSample.y * inv;
    accSample.z = accSample.z * (1.0f - inv) + newSample.z * inv;
    accSample.w = accSample.w * (1.0f - inv) + newSample.w * inv;
    surf2Dwrite(accSample, surfAcc, byteOffset, y);
}

void launchPathTraceKernel(cudaSurfaceObject_t surfNew,
                           cudaSurfaceObject_t surfAcc,
                           const DeviceSceneView& scene,
                           const SceneParams& params) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_params, &params, sizeof(SceneParams)));

    dim3 block(BlockSizeX, BlockSizeY);
    dim3 grid((scene.width + block.x - 1) / block.x,
              (scene.height + block.y - 1) / block.y);
    kernelPathTrace<<<grid, block>>>(surfNew, surfAcc, scene);
    CUDA_CHECK_LAST();
}

} // namespace pt
