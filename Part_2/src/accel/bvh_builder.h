#pragma once

#include "accel/bvh_common.h"
#include "geometry/vertex.h"
#include <vector>

namespace pt {

class BVHBuilder {
public:
    BVH build(const std::vector<Vertex>& vertices,
              std::vector<TriangleFace>& faces,
              uint32_t start, uint32_t end);

private:
    std::vector<BVHNode>      nodes_;
    std::vector<BVHPrimitive> prims_;

    uint32_t buildNode(uint32_t start, uint32_t end);
    uint32_t SAH(uint32_t start, uint32_t end, const float localMin[3], const float localMax[3]);
    uint32_t reorderNodesRecursive(std::vector<BVHNode>& reordered, uint32_t oldIndex);
    void     reorderNodes();
    void     reorderFaces(std::vector<TriangleFace>& faces);

    void getLocalBound(uint32_t start, uint32_t end, float outMin[3], float outMax[3]) const;
    void getCentroidBound(uint32_t start, uint32_t end, float outMin[3], float outMax[3]) const;
};

} // namespace pt
