#pragma once

#include "config.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <float.h>
#include "vec3.h"
#include "Bound.h"
#include "math_utils.h"
#include "Triangle.h"
#include "BVH.h"


BEGIN_NAMESPACE_PT

constexpr uint32_t STACK_DEPTH = 32;

struct HitResult {
    bool isHit;
    float distance;
    int materialID;
    vec3 hitPoint;
    vec3 normal;

    PT_CPU_GPU HitResult() : isHit(false), distance(float_max()), materialID(-1), hitPoint(), normal() {}
};

struct Ray {
    vec3 origin;
    vec3 direction;

    Ray() = default;

    PT_CPU_GPU Ray(const vec3& o, const vec3& d) : origin(o), direction(d.normalize()) {}

    PT_CPU_GPU vec3 at(float t) const { return origin + t * direction; }

    // Version 1: 基础 slab 法，存在分支
    // PT_CPU_GPU bool intersectBound(const Bound& bound) const {
    //     vec3 dir_inv(1.0f / direction[0], 1.0f / direction[1], 1.0f / direction[2]);
    //     float t_max = float_max();
    //     float t_min = float_min();

    //     for (int axis = 0; axis < 3; ++axis) {
    //         if (direction[axis] != 0.0f) {
    //             float t1 = (bound.min[axis] - origin[axis]) * dir_inv[axis];
    //             float t2 = (bound.max[axis] - origin[axis]) * dir_inv[axis];

    //             t_min = max(t_min, min(t1, t2));
    //             t_max = min(t_max, max(t1, t2));
    //         } else if (origin[axis] <= bound.min[axis] || origin[axis] >= bound.max[axis]) {
    //             return false;
    //         }
    //     }
    //     return t_max >= max(t_min, 0.0f);
    // }

    // Verison 2: Branchless Ray/Bounding Box Intersections
    // https://tavianator.com/2015/ray_box_nan.html
    // PT_CPU_GPU bool intersectBound(const Bound& bound) const {
    //     vec3 dir_inv(1.0f / direction[0], 1.0f / direction[1], 1.0f / direction[2]);
    //     float t1 = (bound.min[0] - origin[0]) * dir_inv[0];
    //     float t2 = (bound.max[0] - origin[0]) * dir_inv[0];

    //     float t_min = min(t1, t2);
    //     float t_max = max(t1, t2);

    //     for (int i = 1; i < 3; ++i) {
    //         t1 = (bound.min[i] - origin[i]) * dir_inv[i];
    //         t2 = (bound.max[i] - origin[i]) * dir_inv[i];

    //         t_min = max(t_min, min(t1, t2));
    //         t_max = min(t_max, max(t1, t2));
    //     }

    //     return t_max >= max(t_min, 0.0f);
    // }

    // Version 3: 处理边界情况，性能接近
    // https://tavianator.com/2022/ray_box_boundary.html
    PT_CPU_GPU bool intersectBound(const Bound& bound) const {
        vec3 dir_inv(1.0f / direction[0], 1.0f / direction[1], 1.0f / direction[2]);
        float t_min = 0.0f;
        float t_max = float_max();

        for (int i = 0; i < 3; ++i) {
            float t1 = (bound.min[i] - origin[i]) * dir_inv[i];
            float t2 = (bound.max[i] - origin[i]) * dir_inv[i];

            t_min = min(max(t1, t_min), max(t2, t_min));
            t_max = max(min(t1, t_max), min(t2, t_max));
        }

        return t_min <= t_max;
    }

    // Version 1
    // PT_CPU_GPU HitResult intersectTriangle(const Triangle& triangle) const {
    //     HitResult res;

    //     const vec3& S = this->origin;
    //     const vec3& d = this->direction;
    //     vec3 N        = triangle.normal;

    //     float denom = pt::dot(N, d);

    //     // 若入射方向与平面平行
    //     if (pt::abs(denom) < 0.0001f) return res; 

    //     // 若光线入射方向为三角形背面，将 N 反转
    //     if (denom > 0.0f) {
    //         N     = -N;
    //         denom = -denom;
    //     }

    //     float t = (pt::dot(N, triangle.p1) - pt::dot(S, N)) / denom;
    //     // 若交点太近或在背后
    //     if (t < 0.005) return res;

    //     vec3 P = S + d * t;

    //     // 内部测试
    //     vec3 c1 = pt::cross(triangle.p2 - triangle.p1, P - triangle.p1);
    //     vec3 c2 = pt::cross(triangle.p3 - triangle.p2, P - triangle.p2);
    //     vec3 c3 = pt::cross(triangle.p1 - triangle.p3, P - triangle.p3);

    //     const vec3& n = triangle.normal;
    //     if (pt::dot(c1, n) < 0.0f || pt::dot(c2, n) < 0.0f || pt::dot(c3, n) < 0.0f) return res;

    //     res.isHit      = true;
    //     res.distance   = t;
    //     res.hitPoint   = P;
    //     res.materialID = triangle.materialID;
    //     res.normal     = n;
    //     return res;
    // }

    // Version 2: MT 算法
    PT_CPU_GPU HitResult intersectTriangle(const Triangle& triangle) const {
        HitResult res;

        // 计算三角形的边向量
        vec3 edge1 = triangle.p2 - triangle.p1;
        vec3 edge2 = triangle.p3 - triangle.p1;

        // 计算行列式
        vec3 pvec = cross(direction, edge2);
        float det = dot(edge1, pvec);

        // 判断平行
        if (abs(det) < float_epsilon()) return res;

        float invDet = 1.0f / det;

        // 计算从 p1 到 origin 的向量
        vec3 tvec = origin - triangle.p1;

        // 计算重心坐标 u, 并检验是否在 [0, 1] 内
        float u = dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1) return res;

        // 计算坐标 v
        vec3 qvec = cross(tvec, edge1);
        float v   = dot(direction, qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f) return res;

        // 计算射线参数 t, 这里丢弃了背面的情况
        float t = dot(edge2, qvec) * invDet;
        if (t < 1e-4) return res;

        res.isHit      = true;
        res.distance   = t;
        res.hitPoint   = origin + t * direction;
        res.normal     = normalize((1 - u - v) * triangle.n1 + u * triangle.n2 + v * triangle.n3);
        res.materialID = triangle.materialID;

        return res;
    }

    PT_CPU_GPU HitResult intersectBVH(const BVHNode* nodes, const Triangle* triangles) {
        HitResult res;

        // 使用栈进行深度优先搜索
        uint32_t stack[STACK_DEPTH];
        int stackIndex = 0;

        stack[stackIndex++] = 0; // 将根节点入栈

        while (stackIndex > 0) {
            int nodeIndex       = stack[--stackIndex];
            const BVHNode& node = nodes[nodeIndex];

            // 若与 bound 不相交
            if (!intersectBound(node.bound)) { continue; }

            // 若与 bound 相交，分为两种情况
            // 1. node 为叶子节点，遍历 node 中所有 primitive，获取最近的交点
            if (node.isLeaf) {
                for (uint32_t i = 0; i < node.leaf.primCount; ++i) {
                    uint32_t primIndex = node.leaf.firstPrimIndex + i;
                    HitResult h        = intersectTriangle(triangles[primIndex]);
                    if (h.isHit && h.distance < res.distance) { res = h; }
                }
            } else {
                // 2. node 为内部节点，将左右子树压入栈
                stack[stackIndex++] = node.internal.leftIndex;
                stack[stackIndex++] = node.internal.rightIndex;
            }
        }
        return res;
    }
};

END_NAMESPACE_PT