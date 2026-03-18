#pragma once

#include "geometry/vertex.h"
#include "math/mat4.h"
#include <vector>
#include <string>

namespace pt {

class MeshLoader {
public:
    static bool loadObj(const std::string& filename,
                        int materialID,
                        const mat4& transform,
                        std::vector<Vertex>& outVertices,
                        std::vector<TriangleFace>& outFaces);
};

} // namespace pt
