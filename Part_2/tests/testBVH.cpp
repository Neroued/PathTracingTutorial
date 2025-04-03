#include <iostream>
#include <vector>
#include "BVH.h"
#include "Triangle.h"
#include "Ray.h"

using namespace pt;

// 递归函数：按照树形结构输出BVH
void printBVHRecursive(const BVH& bvh, uint32_t nodeIndex, int level = 0) {
    const BVHNode& node = bvh.nodes[nodeIndex];
    // 使用缩进表示层级
    std::string indent(level * 2, ' ');
    std::cout << indent << (node.isLeaf ? "Leaf" : "Internal") << " Node:" << std::endl;
    std::cout << indent << "  Bound min: (" << node.bound.min[0] << ", " << node.bound.min[1] << ", " << node.bound.min[2] << ")" << std::endl;
    std::cout << indent << "  Bound max: (" << node.bound.max[0] << ", " << node.bound.max[1] << ", " << node.bound.max[2] << ")" << std::endl;

    if (node.isLeaf) {
        std::cout << indent << "  Primitive count: " << node.leaf.primCount << std::endl;
    } else {
        // 递归输出左右子节点
        printBVHRecursive(bvh, node.internal.leftIndex, level + 1);
        printBVHRecursive(bvh, node.internal.rightIndex, level + 1);
    }
}

// 输出整个BVH结构（从根节点开始）
void printBVH(const BVH& bvh) {
    if (bvh.nodes.empty()) {
        std::cout << "BVH is empty!" << std::endl;
        return;
    }
    printBVHRecursive(bvh, 0, 0);
}

// 测试函数：构造场景并检测光线与BVH求交是否正确
void testBVHIntersection() {
    // 构造简单场景：两个三角形
    std::vector<Triangle> triangles;
    // 第一个三角形：位于 z=0 平面，三个顶点分别为 (0,0,0), (1,0,0), (0,1,0)
    triangles.push_back(Triangle(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0)));
    // 第二个三角形：位于 z=1 平面
    triangles.push_back(Triangle(vec3(0, 0, 1), vec3(1, 0, 1), vec3(0, 1, 1)));
    triangles[0].materialID = 0;
    triangles[1].materialID = 1;

    // 构建BVH：假设 BVH::build 能正确构建整个加速结构
    BVH bvh;
    bvh.build(triangles, 0, static_cast<uint32_t>(triangles.size()));

    // 输出BVH结构，便于调试和验证
    std::cout << "BVH Structure:" << std::endl;
    printBVH(bvh);

    // 构造一条光线：起点在 (0.2, 0.2, -1) 处，方向指向 +z 轴
    Ray ray(vec3(0.2f, 0.2f, -1.0f), vec3(0, 0, 1));

    // 求交：调用 BVH 求交函数
    HitResult result = ray.intersectBVH(bvh.nodes.data(), triangles.data());

    // 输出求交结果
    if (result.isHit) {
        std::cout << "Intersection detected:" << std::endl;
        std::cout << "  Distance: " << result.distance << std::endl;
        std::cout << "  Hit Point: (" << result.hitPoint[0] << ", " << result.hitPoint[1] << ", " << result.hitPoint[2] << ")" << std::endl;
        std::cout << "  Normal: (" << result.normal[0] << ", " << result.normal[1] << ", " << result.normal[2] << ")" << std::endl;
        std::cout << "  Material ID: " << result.materialID << std::endl;
    } else {
        std::cout << "No intersection detected." << std::endl;
    }
}

int main() {
    testBVHIntersection();
    return 0;
}
