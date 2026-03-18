#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/math_utils.h"
#include "geometry/vertex.h"
#include "lights/envmap.h"
#include "scene/device_scene.h"

namespace pt {

__device__ inline float environmentPdf(const DeviceSceneView& scene, const vec3& direction) {
    if (scene.hdrTex == 0 || scene.envPmf == nullptr || scene.numEnvTexels == 0 ||
        scene.envWidth <= 0 || scene.envHeight <= 0) {
        return 0.0f;
    }

    vec2 uv = sampleSphericalMap(direction);
    int x = static_cast<int>(pt::clamp(uv.x * scene.envWidth,  0.0f, static_cast<float>(scene.envWidth  - 1)));
    int y = static_cast<int>(pt::clamp(uv.y * scene.envHeight, 0.0f, static_cast<float>(scene.envHeight - 1)));

    float sinTheta = pt::sqrt(pt::max(0.0f, 1.0f - direction.y * direction.y));
    if (sinTheta <= 1e-5f) return 0.0f;

    int idx = y * scene.envWidth + x;
    return scene.envPmf[idx] * static_cast<float>(scene.envWidth * scene.envHeight) / (TwoPi * Pi * sinTheta);
}

__device__ inline float emissiveTrianglePdf(const DeviceSceneView& scene,
                                            int primitiveId,
                                            float distance,
                                            const vec3& direction) {
    if (primitiveId < 0 || primitiveId >= static_cast<int>(scene.numFaces) ||
        scene.triangleLightPmf == nullptr) {
        return 0.0f;
    }

    float selectPmf = scene.triangleLightPmf[primitiveId];
    if (selectPmf <= 0.0f) return 0.0f;

    const TriangleFace& face = scene.faces[primitiveId];
    const Vertex& va = scene.vertices[face.v0];
    const Vertex& vb = scene.vertices[face.v1];
    const Vertex& vc = scene.vertices[face.v2];

    float area = faceArea(va, vb, vc);
    if (area <= 0.0f) return 0.0f;

    vec3 lightNormal = faceGeometricNormal(va, vb, vc);
    float cosLight   = pt::abs(dot(lightNormal, -direction));
    if (cosLight <= 1e-5f) return 0.0f;

    return selectPmf * distance * distance / (area * cosLight);
}

} // namespace pt
