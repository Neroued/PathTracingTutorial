#include "geometry/shapes.h"

namespace pt::shapes {

void buildQuad(const vec3& a, const vec3& b, const vec3& c, const vec3& d,
               int matId,
               std::vector<Vertex>& outVertices,
               std::vector<TriangleFace>& outFaces)
{
    vec3 normal = normalize(cross(b - a, c - a));
    vec3 tangentDir = normalize(b - a);
    vec4 tangent(tangentDir.x, tangentDir.y, tangentDir.z, 1.0f);
    uint32_t base = static_cast<uint32_t>(outVertices.size());

    outVertices.push_back({a, normal, {0.0f, 0.0f}, tangent});
    outVertices.push_back({b, normal, {1.0f, 0.0f}, tangent});
    outVertices.push_back({c, normal, {1.0f, 1.0f}, tangent});
    outVertices.push_back({d, normal, {0.0f, 1.0f}, tangent});

    outFaces.push_back({base + 0, base + 1, base + 2, matId});
    outFaces.push_back({base + 0, base + 2, base + 3, matId});
}

} // namespace pt::shapes
