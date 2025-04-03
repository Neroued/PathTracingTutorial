#pragma once

#include "config.h"
#include <cuda_runtime.h>
#include "vec3.h"
#include "math_utils.h"

BEGIN_NAMESPACE_PT

struct alignas(16) Bound {
    vec3 min;
    vec3 max;

    PT_CPU_GPU Bound() { reset(); }

    PT_CPU_GPU Bound(const Bound& other) : max(other.max), min(other.min) {}

    PT_CPU_GPU void reset() {
        min = vec3(pt::float_max(), pt::float_max(), pt::float_max());
        max = -min;
    }

    PT_CPU_GPU void expand(const vec3& point) {
        min = minVec(min, point);
        max = maxVec(max, point);
    }

    PT_CPU_GPU void expand(const Bound& other) {
        min = minVec(min, other.min);
        max = maxVec(max, other.max);
    }

    // 计算 Bound 表面积
    PT_CPU_GPU float surfaceArea() const {
        // 若 Bound 非法，返回 0
        if (max.x < min.x || max.y < min.y || max.z < min.z) return 0.0f;
        vec3 d = diagonal();
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    // 对角线向量
    PT_CPU_GPU vec3 diagonal() const { return max - min; }

    // 对角线长度最大的维度索引
    PT_CPU_GPU int maxDimension() const {
        vec3 d = diagonal();
        if (d.x > d.y && d.x > d.z) {
            return 0;
        } else if (d.y > d.z) {
            return 1;
        } else {
            return 2;
        }
    }

    // v 在 Bound 内的归一化偏移
    PT_CPU_GPU vec3 offset(const vec3& v) const {
        vec3 o = v - min;
        vec3 d = diagonal();
        return vec3((d.x > 0.0f) ? o.x / d.x : 0.0f, (d.y > 0.0f) ? o.y / d.y : 0.0f, (d.z > 0.0f) ? o.z / d.z : 0.0f);
    }

    PT_CPU_GPU vec3 centroid() const { return 0.5f * max + 0.5f * min; }

    PT_CPU_GPU static Bound Union(const Bound& a, const Bound& b) {
        Bound res(a);
        res.expand(b);
        return res;
    }

private:
    PT_CPU_GPU vec3 minVec(const vec3& a, const vec3& b) { return vec3(pt::min(a.x, b.x), pt::min(a.y, b.y), pt::min(a.z, b.z)); }

    PT_CPU_GPU vec3 maxVec(const vec3& a, const vec3& b) { return vec3(pt::max(a.x, b.x), pt::max(a.y, b.y), pt::max(a.z, b.z)); }
};

END_NAMESPACE_PT