#pragma once

#include "geometry/vertex.h"
#include "math/vecmath.h"

#include <vector>

namespace pt::shapes {

void buildQuad(const vec3& a, const vec3& b, const vec3& c, const vec3& d,
               int matId,
               std::vector<Vertex>& outVertices,
               std::vector<TriangleFace>& outFaces);

} // namespace pt::shapes
