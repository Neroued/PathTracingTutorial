#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "geometry/vertex.h"
#include "materials/material.h"
#include "io/image_io.h"

#include <vector>

namespace pt {

struct alignas(16) EmissiveTriangleRef {
    uint32_t triangleIndex = 0;
    float    selectionPmf = 0.0f;
    float    cdf          = 0.0f;
    float    pad          = 0.0f;
};

struct alignas(8) AliasEntry {
    float    prob;
    uint32_t alias;
};

struct HostLightSampler {
    std::vector<EmissiveTriangleRef> emissiveTriangles;
    std::vector<float>               triangleSelectionPmf;

    std::vector<AliasEntry>          triangleAlias;

    std::vector<float>               envPmf;
    std::vector<AliasEntry>          envAlias;
    int                              envWidth  = 0;
    int                              envHeight = 0;

    bool hasEmissiveTriangles() const { return !emissiveTriangles.empty(); }
    bool hasEnvironment() const {
        return envWidth > 0 && envHeight > 0 && !envPmf.empty() && !envAlias.empty();
    }
};

PT_HD inline float luminance(const vec3& color) {
    return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}

PT_HD inline vec3 faceGeometricNormal(const Vertex& v0, const Vertex& v1, const Vertex& v2) {
    return normalize(cross(v1.position - v0.position, v2.position - v0.position));
}

PT_HD inline float faceArea(const Vertex& v0, const Vertex& v1, const Vertex& v2) {
    return 0.5f * cross(v1.position - v0.position, v2.position - v0.position).length();
}

PT_HD inline vec3 emittedRadiance(const Material& mat) {
    return mat.emissive * mat.emissiveStrength;
}

PT_HD inline void sampleTriangleSurface(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                                        float u1, float u2,
                                        vec3& position, vec3& normal) {
    float su0 = pt::sqrt(u1);
    float b0  = 1.0f - su0;
    float b1  = u2 * su0;
    float b2  = 1.0f - b0 - b1;

    position = v0.position * b0 + v1.position * b1 + v2.position * b2;
    normal   = normalize(v0.normal * b0 + v1.normal * b1 + v2.normal * b2);
    if (normal.length() == 0.0f) {
        normal = faceGeometricNormal(v0, v1, v2);
    }
}

void buildEmissiveTriangleSampler(const std::vector<Vertex>& vertices,
                                  const std::vector<TriangleFace>& faces,
                                  const std::vector<Material>& materials,
                                  HostLightSampler& sampler);

void buildEnvironmentSampler(const Image& hdrImage, HostLightSampler& sampler);

} // namespace pt
