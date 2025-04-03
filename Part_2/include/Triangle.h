#pragma once

#include "config.h"
#include <cuda_runtime.h>
#include "vec3.h"
#include "Bound.h"

BEGIN_NAMESPACE_PT

struct alignas(16) Triangle {
    vec3 p1;
    vec3 p2;
    vec3 p3;
    vec3 normal;
    int materialID;

    Triangle() = default;

    Triangle(const vec3& P1, const vec3& P2, const vec3& P3) : p1(P1), p2(P2), p3(P3) { normal = normalize(cross(p2 - p1, p3 - p1)); }

    Triangle(const Triangle& other) : p1(other.p1), p2(other.p2), p3(other.p3), normal(other.normal), materialID(other.materialID) {}

    Triangle& operator=(const Triangle& other) {
        if (this != &other) {
            p1         = other.p1;
            p2         = other.p2;
            p3         = other.p3;
            normal     = other.normal;
            materialID = other.materialID;
        }
        return *this;
    }

    Triangle(Triangle&& other) noexcept
        : p1(std::move(other.p1)), p2(std::move(other.p2)), p3(std::move(other.p3)), normal(std::move(other.normal)), materialID(other.materialID) {}

    Triangle& operator=(Triangle&& other) noexcept {
        if (this != &other) {
            p1         = std::move(other.p1);
            p2         = std::move(other.p2);
            p3         = std::move(other.p3);
            normal     = std::move(other.normal);
            materialID = other.materialID;
        }
        return *this;
    }

    PT_CPU_GPU Bound getBound() const {
        Bound res;
        res.expand(p1);
        res.expand(p2);
        res.expand(p3);
        return res;
    }
};

END_NAMESPACE_PT