#pragma once

#include "pt.h"
#include "core/ray.h"
#include "core/interaction.h"
#include "math/vecmath.h"
#include "math/math_utils.h"
#include "geometry/vertex.h"
#include "geometry/triangle.h"
#include "accel/bvh_common.h"

namespace pt {

constexpr uint32_t BvhStackDepth = 32;

struct RayInverse {
    vec3  invDir;
    int   sign[3];

    PT_HD explicit RayInverse(const vec3& dir)
        : invDir(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z)
    {
        sign[0] = invDir.x < 0.0f ? 1 : 0;
        sign[1] = invDir.y < 0.0f ? 1 : 0;
        sign[2] = invDir.z < 0.0f ? 1 : 0;
    }
};

PT_HD inline bool intersectBounds(const Ray& ray, const RayInverse& ri,
                                  const BVHNode& node, float tMax) {
    float t_min = 0.0f;
    float t_max = tMax;

    for (int i = 0; i < 3; ++i) {
        float t1 = (node.bmin[i] - ray.origin[i]) * ri.invDir[i];
        float t2 = (node.bmax[i] - ray.origin[i]) * ri.invDir[i];
        t_min = pt::min(pt::max(t1, t_min), pt::max(t2, t_min));
        t_max = pt::max(pt::min(t1, t_max), pt::min(t2, t_max));
    }
    return t_min <= t_max;
}

PT_HD inline SurfaceInteraction intersectScene(
    const Ray&           ray,
    const BVHNode*       nodes,
    const Vertex*        vertices,
    const TriangleFace*  faces)
{
    SurfaceInteraction result;
    RayInverse ri(ray.direction);

    uint32_t stack[BvhStackDepth];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        uint32_t idx        = stack[--top];
        const BVHNode& node = nodes[idx];

        if (!intersectBounds(ray, ri, node, result.distance)) continue;

        if (node.isLeaf()) {
            uint32_t first = node.firstPrimIndex();
            uint32_t count = node.primCount();
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t primIdx = first + i;
                const TriangleFace& face = faces[primIdx];
                SurfaceInteraction si = intersectTriangle(
                    ray,
                    vertices[face.v0], vertices[face.v1], vertices[face.v2],
                    face.materialID);
                if (si.hit && si.distance < result.distance) {
                    si.primitiveId = static_cast<int>(primIdx);
                    result = si;
                }
            }
        } else {
            uint32_t left  = node.leftIndex();
            uint32_t right = node.rightIndex();

            int axis = node.maxDimension();
            float cL = nodes[left].centroid(axis);
            float cR = nodes[right].centroid(axis);
            bool flipOrder = (ri.sign[axis] != 0);

            if (flipOrder == (cL < cR)) {
                stack[top++] = left;
                stack[top++] = right;
            } else {
                stack[top++] = right;
                stack[top++] = left;
            }
        }
    }
    return result;
}

PT_HD inline bool occludedScene(
    const Ray&           ray,
    const BVHNode*       nodes,
    const Vertex*        vertices,
    const TriangleFace*  faces,
    float                tMax)
{
    RayInverse ri(ray.direction);

    uint32_t stack[BvhStackDepth];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        uint32_t idx        = stack[--top];
        const BVHNode& node = nodes[idx];

        if (!intersectBounds(ray, ri, node, tMax)) continue;

        if (node.isLeaf()) {
            uint32_t first = node.firstPrimIndex();
            uint32_t count = node.primCount();
            for (uint32_t i = 0; i < count; ++i) {
                const TriangleFace& face = faces[first + i];
                float t = intersectTriangleShadow(
                    ray,
                    vertices[face.v0], vertices[face.v1], vertices[face.v2]);
                if (t > 0.0f && t < tMax) return true;
            }
        } else {
            stack[top++] = node.leftIndex();
            stack[top++] = node.rightIndex();
        }
    }
    return false;
}

} // namespace pt
