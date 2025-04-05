#include "Bound.h"
#include "config.h"
#include "BVH.h"
#include "Triangle.h"
#include "vector"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <future>

BEGIN_NAMESPACE_PT

constexpr int LEAF_THRESHOLD     = 2;
constexpr int PARALLEL_THRESHOLD = 1024;

constexpr int numBuckets = 24;
constexpr int numSplits  = numBuckets - 1;

std::atomic<uint32_t> nodeCount = 0;

void BVH::build(std::vector<Triangle>& triangles, uint32_t start, uint32_t end) {
    // 根据输入的三角形数组填充 bounds 和 triIndices
    prims.clear();
    prims.resize(triangles.size());

    for (size_t i = 0; i < triangles.size(); ++i) { prims[i] = BVHPrimitive(triangles[i].getBound(), i); }

    // 预留节点空间
    nodes.resize(2 * triangles.size());

    // 递归建树
    buildNode(0, triangles.size());

    // 调整为合适大小
    nodes.resize(nodeCount.load());

    // 将节点转为广度优先顺序
    reorderNodes();

    // 按照 prims.index 重排 triangles
    reorderTriangles(triangles);

    // 清理不需要的空间
    std::vector<BVHPrimitive>().swap(prims);
}

uint32_t BVH::buildNode(uint32_t start, uint32_t end) {
    uint32_t nodeIndex = nodeCount.fetch_add(1, std::memory_order_relaxed);
    BVHNode& node      = nodes[nodeIndex];

    // 计算当前范围内的包围盒
    Bound localBound = getLocalBound(start, end);
    node.bound       = localBound;

    // 判断是否建立叶子节点
    int primCount = end - start;
    if (primCount < LEAF_THRESHOLD) {
        node.leaf.firstPrimIndex = start;
        node.leaf.primCount      = primCount;
        node.isLeaf              = true;

        return nodeIndex;
    }

    // 使用 SAH 寻找最佳分割点，同时会对 prims 进行重排
    uint32_t mid = SAH(start, end, localBound);

    // SAH 认为应当生成叶子节点
    if (mid == end) {
        node.leaf.firstPrimIndex = start;
        node.leaf.primCount      = primCount;
        node.isLeaf              = true;

        return nodeIndex;
    }

    // 多线程并行构造
    uint32_t leftIndex, rightIndex;
    if (primCount > PARALLEL_THRESHOLD) {
        auto futureLeft = std::async(std::launch::async, &BVH::buildNode, this, start, mid);
        rightIndex      = buildNode(mid, end);
        leftIndex       = futureLeft.get();
    } else {
        leftIndex  = buildNode(start, mid);
        rightIndex = buildNode(mid, end);
    }
    node.internal.leftIndex  = leftIndex;
    node.internal.rightIndex = rightIndex;
    node.isLeaf              = false;

    return nodeIndex;
}

uint32_t BVH::SAH(uint32_t start, uint32_t end, const Bound& localBound) {
    // 计算区间内包围盒中心点的包围盒
    Bound centroidBound = getCentroidBound(start, end);

    // 选取最大轴对应的维度
    int dim = centroidBound.maxDimension();

    // 根据最大维度将 Primitive 装入桶
    BVHBucket buckets[numBuckets];

    auto computeBucketIndex = [&](const BVHPrimitive& prim) -> int {
        int b = numBuckets * centroidBound.offset(prim.bound.centroid())[dim];
        if (b == numBuckets) b = numBuckets - 1;
        return b;
    };

    auto addToBucket = [&](int b, const BVHPrimitive& prim) {
        buckets[b].count++;
        buckets[b].bound.expand(prim.bound);
    };

    for (uint32_t i = start; i < end; ++i) {
        int b = computeBucketIndex(prims[i]);
        addToBucket(b, prims[i]);
    }

    // 计算在各个分割点处的 cost
    // 计算公式为 count * surfaceArea
    float costs[numSplits] = {};

    // 1. forward
    uint32_t countForward = 0;
    Bound boundForward;
    for (int i = 0; i < numSplits; ++i) {
        countForward += buckets[i].count;
        boundForward.expand(buckets[i].bound);
        costs[i] += countForward * boundForward.surfaceArea();
    }

    // 2. backward
    uint32_t countBackward = 0;
    Bound boundBackward;
    for (int i = numSplits - 1; i >= 0; --i) {
        countBackward += buckets[i + 1].count;
        boundBackward.expand(buckets[i + 1].bound);
        costs[i] += countBackward * boundBackward.surfaceArea();
    }

    // 3. find minimum cost
    auto minIt        = std::min_element(costs, costs + numSplits);
    float minCost     = *minIt;
    int minCostBucket = static_cast<int>(minIt - costs); // 表示在这个桶与下一个桶之间进行分割

    // 比较生成叶子节点与进行分割的代价
    float leafCost = end - start;                               // 假设代价为图元数量
    minCost        = 0.5f + minCost / localBound.surfaceArea(); // 将最小代价转化为相对代价

    // 选择分割或是生成叶子节点
    auto shouldGoLeft = [&](const BVHPrimitive& prim) {
        int b = computeBucketIndex(prim);
        return b <= minCostBucket;
    };

    if (leafCost < minCost) {
        return end;
    } else {
        auto midIt = std::partition(prims.begin() + start, prims.begin() + end, shouldGoLeft);
        return static_cast<uint32_t>(midIt - prims.begin());
    }
}

// 返回节点在排序后数组中的位置
uint32_t BVH::reorderNodesRecursive(std::vector<BVHNode>& reordered, uint32_t oldIndex) {
    // oldIndex 表示原始数据中的节点位置
    reordered.push_back(nodes[oldIndex]);
    uint32_t currentIndex = reordered.size() - 1;

    if (!nodes[oldIndex].isLeaf) {
        reordered[currentIndex].internal.leftIndex  = reorderNodesRecursive(reordered, nodes[oldIndex].internal.leftIndex);
        reordered[currentIndex].internal.rightIndex = reorderNodesRecursive(reordered, nodes[oldIndex].internal.rightIndex);
    }

    // 返回移动后节点的位置
    return currentIndex;
}

void BVH::reorderNodes() {
    std::vector<BVHNode> reordered;
    reordered.reserve(nodes.size());

    // 深度优先，递归将节点存储新数组
    reorderNodesRecursive(reordered, 0);

    nodes.swap(reordered);
}

void BVH::reorderTriangles(std::vector<Triangle>& triangles) {
    std::vector<Triangle> reordered(triangles.size());

    for (uint32_t i = 0; i < triangles.size(); ++i) { reordered[i] = triangles[prims[i].index]; }

    triangles.swap(reordered);
}

Bound BVH::getLocalBound(uint32_t start, uint32_t end) const {
    Bound bound;
    for (uint32_t i = start; i < end; ++i) { bound.expand(prims[i].bound); }
    return bound;
}

Bound BVH::getCentroidBound(uint32_t start, uint32_t end) const {
    Bound centroidBound;
    for (uint32_t i = start; i < end; ++i) { centroidBound.expand(prims[i].bound.centroid()); }
    return centroidBound;
}

END_NAMESPACE_PT