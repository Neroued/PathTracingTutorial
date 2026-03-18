#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/math_utils.h"
#include <cstdint>
#include <vector>

namespace pt {

struct Vertex;
struct TriangleFace;

static constexpr uint32_t BVH_LEAF_FLAG = 0x80000000u;

struct BVHNode {
    float    bmin[3];
    uint32_t data0;
    float    bmax[3];
    uint32_t data1;

    PT_HD bool isLeaf() const { return (data0 & BVH_LEAF_FLAG) != 0; }

    PT_HD uint32_t leftIndex()      const { return data0; }
    PT_HD uint32_t rightIndex()     const { return data1; }
    PT_HD uint32_t firstPrimIndex() const { return data0 & ~BVH_LEAF_FLAG; }
    PT_HD uint32_t primCount()      const { return data1; }

    PT_HD float surfaceArea() const {
        float dx = bmax[0] - bmin[0];
        float dy = bmax[1] - bmin[1];
        float dz = bmax[2] - bmin[2];
        return 2.0f * (dx * dy + dx * dz + dy * dz);
    }

    PT_HD int maxDimension() const {
        float dx = bmax[0] - bmin[0];
        float dy = bmax[1] - bmin[1];
        float dz = bmax[2] - bmin[2];
        if (dx > dy && dx > dz) return 0;
        return dy > dz ? 1 : 2;
    }

    PT_HD float centroid(int axis) const {
        return 0.5f * (bmin[axis] + bmax[axis]);
    }

    static BVHNode makeLeaf(const float lo[3], const float hi[3],
                            uint32_t first, uint32_t count) {
        BVHNode n;
        for (int i = 0; i < 3; ++i) { n.bmin[i] = lo[i]; n.bmax[i] = hi[i]; }
        n.data0 = first | BVH_LEAF_FLAG;
        n.data1 = count;
        return n;
    }

    static BVHNode makeInternal(const float lo[3], const float hi[3],
                                uint32_t left, uint32_t right) {
        BVHNode n;
        for (int i = 0; i < 3; ++i) { n.bmin[i] = lo[i]; n.bmax[i] = hi[i]; }
        n.data0 = left;
        n.data1 = right;
        return n;
    }
};

struct BVHPrimitive {
    float    bmin[3];
    float    bmax[3];
    uint32_t index;

    BVHPrimitive() = default;

    BVHPrimitive(const float lo[3], const float hi[3], uint32_t idx) : index(idx) {
        for (int i = 0; i < 3; ++i) { bmin[i] = lo[i]; bmax[i] = hi[i]; }
    }

    float centroid(int axis) const { return 0.5f * (bmin[axis] + bmax[axis]); }
};

struct BVHBucket {
    float    bmin[3];
    float    bmax[3];
    uint32_t count = 0;

    BVHBucket() {
        for (int i = 0; i < 3; ++i) {
            bmin[i] = float_max();
            bmax[i] = -float_max();
        }
    }

    void expand(const BVHPrimitive& p) {
        for (int i = 0; i < 3; ++i) {
            bmin[i] = pt::min(bmin[i], p.bmin[i]);
            bmax[i] = pt::max(bmax[i], p.bmax[i]);
        }
    }

    float surfaceArea() const {
        float dx = bmax[0] - bmin[0];
        float dy = bmax[1] - bmin[1];
        float dz = bmax[2] - bmin[2];
        if (dx < 0 || dy < 0 || dz < 0) return 0.0f;
        return 2.0f * (dx * dy + dx * dz + dy * dz);
    }
};

struct BVH {
    std::vector<BVHNode> nodes;
};

} // namespace pt
