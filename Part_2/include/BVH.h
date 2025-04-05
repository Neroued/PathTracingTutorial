#pragma once

#include "config.h"
#include "Bound.h"
#include <cstdint>
#include <vector>

BEGIN_NAMESPACE_PT

struct Triangle;

struct alignas(16) BVHNode {
    Bound bound; /* 32 bytes */

    union {
        struct {
            uint32_t leftIndex, rightIndex;
        } internal;

        struct {
            uint32_t firstPrimIndex, primCount;
        } leaf;
    }; /* 8 bytes */

    bool isLeaf;
};

struct BVHPrimitive {
    Bound bound;
    uint32_t index;

    BVHPrimitive() {}

    BVHPrimitive(const Bound& bd, uint32_t idx) : bound(bd), index(idx) {}
};

struct BVHBucket {
    Bound bound;
    uint32_t count = 0;
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<BVHPrimitive> prims;

    // 在节点范围 [start, end) 之间建立 BVH
    void build(std::vector<Triangle>& triangles, uint32_t start, uint32_t end);

private:
    uint32_t buildNode(uint32_t start, uint32_t end);

    // Surface Area Heuristic 寻找最佳分割位置
    uint32_t SAH(uint32_t start, uint32_t end, const Bound& localBound);

    uint32_t reorderNodesRecursive(std::vector<BVHNode>& reordered, uint32_t oldIndex);
    void reorderNodes();                                     // 将节点变为深度优先排序
    void reorderTriangles(std::vector<Triangle>& triangles); // 按照 prims.index 重排 triangles

    Bound getLocalBound(uint32_t start, uint32_t end) const;
    Bound getCentroidBound(uint32_t start, uint32_t end) const;
};

END_NAMESPACE_PT