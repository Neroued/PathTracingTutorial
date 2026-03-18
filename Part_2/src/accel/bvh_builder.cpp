#include "accel/bvh_builder.h"
#include "geometry/vertex.h"

#include <algorithm>
#include <atomic>
#include <future>

namespace pt {

constexpr int LeafThreshold     = 4;
constexpr int ParallelThreshold = 1024;
constexpr int NumBuckets        = 24;
constexpr int NumSplits         = NumBuckets - 1;

static std::atomic<uint32_t> s_nodeCount{0};

static void boundsReset(float lo[3], float hi[3]) {
    lo[0] = lo[1] = lo[2] =  float_max();
    hi[0] = hi[1] = hi[2] = -float_max();
}

static void boundsExpand(float lo[3], float hi[3], const float plo[3], const float phi[3]) {
    for (int i = 0; i < 3; ++i) {
        lo[i] = pt::min(lo[i], plo[i]);
        hi[i] = pt::max(hi[i], phi[i]);
    }
}

static void boundsExpandPoint(float lo[3], float hi[3], float x, float y, float z) {
    lo[0] = pt::min(lo[0], x); hi[0] = pt::max(hi[0], x);
    lo[1] = pt::min(lo[1], y); hi[1] = pt::max(hi[1], y);
    lo[2] = pt::min(lo[2], z); hi[2] = pt::max(hi[2], z);
}

static float boundsSurfaceArea(const float lo[3], const float hi[3]) {
    float dx = hi[0] - lo[0];
    float dy = hi[1] - lo[1];
    float dz = hi[2] - lo[2];
    if (dx < 0 || dy < 0 || dz < 0) return 0.0f;
    return 2.0f * (dx * dy + dx * dz + dy * dz);
}

static int boundsMaxDimension(const float lo[3], const float hi[3]) {
    float dx = hi[0] - lo[0];
    float dy = hi[1] - lo[1];
    float dz = hi[2] - lo[2];
    if (dx > dy && dx > dz) return 0;
    return dy > dz ? 1 : 2;
}

static float boundsOffset(const float lo[3], const float hi[3], float val, int axis) {
    float d = hi[axis] - lo[axis];
    return d > 0 ? (val - lo[axis]) / d : 0.0f;
}

BVH BVHBuilder::build(const std::vector<Vertex>& vertices,
                      std::vector<TriangleFace>& faces,
                      uint32_t start, uint32_t end) {
    prims_.clear();
    prims_.resize(faces.size());
    for (size_t i = 0; i < faces.size(); ++i) {
        const TriangleFace& f = faces[i];
        const vec3& p0 = vertices[f.v0].position;
        const vec3& p1 = vertices[f.v1].position;
        const vec3& p2 = vertices[f.v2].position;

        BVHPrimitive& bp = prims_[i];
        bp.index = static_cast<uint32_t>(i);
        bp.bmin[0] = pt::min(pt::min(p0.x, p1.x), p2.x);
        bp.bmin[1] = pt::min(pt::min(p0.y, p1.y), p2.y);
        bp.bmin[2] = pt::min(pt::min(p0.z, p1.z), p2.z);
        bp.bmax[0] = pt::max(pt::max(p0.x, p1.x), p2.x);
        bp.bmax[1] = pt::max(pt::max(p0.y, p1.y), p2.y);
        bp.bmax[2] = pt::max(pt::max(p0.z, p1.z), p2.z);
    }

    nodes_.resize(2 * faces.size());
    s_nodeCount.store(0);

    buildNode(0, static_cast<uint32_t>(faces.size()));
    nodes_.resize(s_nodeCount.load());

    reorderNodes();
    reorderFaces(faces);

    BVH result;
    result.nodes = std::move(nodes_);
    prims_.clear();
    return result;
}

uint32_t BVHBuilder::buildNode(uint32_t start, uint32_t end) {
    uint32_t nodeIndex = s_nodeCount.fetch_add(1, std::memory_order_relaxed);

    float localMin[3], localMax[3];
    getLocalBound(start, end, localMin, localMax);

    int primCount = end - start;
    if (primCount < LeafThreshold) {
        nodes_[nodeIndex] = BVHNode::makeLeaf(localMin, localMax, start, primCount);
        return nodeIndex;
    }

    uint32_t mid = SAH(start, end, localMin, localMax);
    if (mid == end) {
        nodes_[nodeIndex] = BVHNode::makeLeaf(localMin, localMax, start, primCount);
        return nodeIndex;
    }

    uint32_t leftIndex, rightIndex;
    if (primCount > ParallelThreshold) {
        auto futureLeft = std::async(std::launch::async, &BVHBuilder::buildNode, this, start, mid);
        rightIndex      = buildNode(mid, end);
        leftIndex       = futureLeft.get();
    } else {
        leftIndex  = buildNode(start, mid);
        rightIndex = buildNode(mid, end);
    }

    nodes_[nodeIndex] = BVHNode::makeInternal(localMin, localMax, leftIndex, rightIndex);
    return nodeIndex;
}

uint32_t BVHBuilder::SAH(uint32_t start, uint32_t end,
                         const float localMin[3], const float localMax[3]) {
    float centMin[3], centMax[3];
    getCentroidBound(start, end, centMin, centMax);
    int dim = boundsMaxDimension(centMin, centMax);

    BVHBucket buckets[NumBuckets];

    auto bucketIndex = [&](const BVHPrimitive& p) -> int {
        float c = p.centroid(dim);
        int b = static_cast<int>(NumBuckets * boundsOffset(centMin, centMax, c, dim));
        return b >= NumBuckets ? NumBuckets - 1 : b;
    };

    for (uint32_t i = start; i < end; ++i) {
        int b = bucketIndex(prims_[i]);
        buckets[b].count++;
        buckets[b].expand(prims_[i]);
    }

    float costs[NumSplits] = {};

    uint32_t countFwd = 0;
    float fwdMin[3], fwdMax[3];
    boundsReset(fwdMin, fwdMax);
    for (int i = 0; i < NumSplits; ++i) {
        countFwd += buckets[i].count;
        boundsExpand(fwdMin, fwdMax, buckets[i].bmin, buckets[i].bmax);
        costs[i] += countFwd * boundsSurfaceArea(fwdMin, fwdMax);
    }

    uint32_t countBwd = 0;
    float bwdMin[3], bwdMax[3];
    boundsReset(bwdMin, bwdMax);
    for (int i = NumSplits - 1; i >= 0; --i) {
        countBwd += buckets[i + 1].count;
        boundsExpand(bwdMin, bwdMax, buckets[i + 1].bmin, buckets[i + 1].bmax);
        costs[i] += countBwd * boundsSurfaceArea(bwdMin, bwdMax);
    }

    auto minIt    = std::min_element(costs, costs + NumSplits);
    float minCost = *minIt;
    int minBucket = static_cast<int>(minIt - costs);

    float leafCost = static_cast<float>(end - start);
    float localSA  = boundsSurfaceArea(localMin, localMax);
    minCost        = 0.5f + (localSA > 0.0f ? minCost / localSA : 0.0f);

    if (leafCost < minCost) return end;

    auto midIt = std::partition(prims_.begin() + start, prims_.begin() + end,
                                [&](const BVHPrimitive& p) { return bucketIndex(p) <= minBucket; });
    return static_cast<uint32_t>(midIt - prims_.begin());
}

uint32_t BVHBuilder::reorderNodesRecursive(std::vector<BVHNode>& reordered, uint32_t oldIndex) {
    reordered.push_back(nodes_[oldIndex]);
    uint32_t cur = static_cast<uint32_t>(reordered.size()) - 1;
    if (!nodes_[oldIndex].isLeaf()) {
        reordered[cur].data0 = reorderNodesRecursive(reordered, nodes_[oldIndex].leftIndex());
        reordered[cur].data1 = reorderNodesRecursive(reordered, nodes_[oldIndex].rightIndex());
    }
    return cur;
}

void BVHBuilder::reorderNodes() {
    std::vector<BVHNode> reordered;
    reordered.reserve(nodes_.size());
    reorderNodesRecursive(reordered, 0);
    nodes_.swap(reordered);
}

void BVHBuilder::reorderFaces(std::vector<TriangleFace>& faces) {
    std::vector<TriangleFace> reordered(faces.size());
    for (uint32_t i = 0; i < faces.size(); ++i)
        reordered[i] = faces[prims_[i].index];
    faces.swap(reordered);
}

void BVHBuilder::getLocalBound(uint32_t start, uint32_t end,
                               float outMin[3], float outMax[3]) const {
    boundsReset(outMin, outMax);
    for (uint32_t i = start; i < end; ++i)
        boundsExpand(outMin, outMax, prims_[i].bmin, prims_[i].bmax);
}

void BVHBuilder::getCentroidBound(uint32_t start, uint32_t end,
                                  float outMin[3], float outMax[3]) const {
    boundsReset(outMin, outMax);
    for (uint32_t i = start; i < end; ++i) {
        float cx = prims_[i].centroid(0);
        float cy = prims_[i].centroid(1);
        float cz = prims_[i].centroid(2);
        boundsExpandPoint(outMin, outMax, cx, cy, cz);
    }
}

} // namespace pt
