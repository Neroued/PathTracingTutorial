#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/math_utils.h"
#include "math/frame.h"
#include "materials/material.h"
#include "geometry/vertex.h"
#include "core/interaction.h"
#include "scene/device_scene.h"

#include <cuda_runtime.h>
#include <texture_indirect_functions.h>

namespace pt {

__device__ inline vec2 applyUvTransform(const Material& mat, const vec2& uv) {
    if (mat.uvRotation == 0.0f && mat.uvScaleX == 1.0f && mat.uvScaleY == 1.0f &&
        mat.uvOffsetX == 0.0f && mat.uvOffsetY == 0.0f)
        return uv;
    float cosR = pt::cos(mat.uvRotation);
    float sinR = pt::sin(mat.uvRotation);
    float ru = cosR * uv.x + sinR * uv.y;
    float rv = -sinR * uv.x + cosR * uv.y;
    return vec2(ru * mat.uvScaleX + mat.uvOffsetX,
                rv * mat.uvScaleY + mat.uvOffsetY);
}

__device__ inline vec4 interpolateTangent(const DeviceSceneView& scene,
                                           const SurfaceInteraction& si) {
    const TriangleFace& face = scene.faces[si.primitiveId];
    float w = 1.0f - si.baryU - si.baryV;
    vec4 t0 = scene.vertices[face.v0].tangent;
    vec4 t1 = scene.vertices[face.v1].tangent;
    vec4 t2 = scene.vertices[face.v2].tangent;
    vec3 t = normalize(vec3(w * t0.x + si.baryU * t1.x + si.baryV * t2.x,
                            w * t0.y + si.baryU * t1.y + si.baryV * t2.y,
                            w * t0.z + si.baryU * t1.z + si.baryV * t2.z));
    float sign = (t0.w >= 0.0f) ? 1.0f : -1.0f;
    return vec4(t.x, t.y, t.z, sign);
}

__device__ inline vec3 applyNormalMap(const Material& mat,
                                      const vec3& N, const vec4& meshTangent,
                                      const vec2& uv,
                                      const DeviceSceneView& scene) {
    if (mat.normalTexId < 0 || scene.textures == nullptr ||
        static_cast<uint32_t>(mat.normalTexId) >= scene.numTextures) {
        return N;
    }
    float4 nm = tex2D<float4>(scene.textures[mat.normalTexId], uv.x, uv.y);
    vec3 mapN(nm.x * 2.0f - 1.0f, nm.y * 2.0f - 1.0f, nm.z * 2.0f - 1.0f);
    mapN.x *= mat.normalScale;
    mapN.y *= mat.normalScale;
    mapN = normalize(mapN);

    vec3 T(meshTangent.x, meshTangent.y, meshTangent.z);
    bool hasMeshTangent = (T.x != 0.0f || T.y != 0.0f || T.z != 0.0f);

    if (hasMeshTangent) {
        T = normalize(T - N * dot(N, T));
        vec3 B = cross(N, T) * meshTangent.w;
        return normalize(T * mapN.x + B * mapN.y + N * mapN.z);
    }

    Frame tbn(N);
    return normalize(tbn.toWorld(mapN));
}

} // namespace pt
