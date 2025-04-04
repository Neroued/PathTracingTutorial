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
    vec3 n1;
    vec3 n2;
    vec3 n3;
    int materialID;

    Triangle() = default;

    Triangle(const vec3& P1, const vec3& P2, const vec3& P3) : p1(P1), p2(P2), p3(P3) {
        n1 = normalize(cross(p2 - p1, p3 - p1));
        n3 = n2 = n1;
    }

    Triangle(const vec3& P1, const vec3& P2, const vec3& P3, const vec3& N1, const vec3& N2, const vec3& N3)
        : p1(P1), p2(P2), p3(P3), n1(N1.normalize()), n2(N2.normalize()), n3(N3.normalize()) {}

    Triangle(const Triangle& other)
        : p1(other.p1), p2(other.p2), p3(other.p3), n1(other.n1), n2(other.n2), n3(other.n3), materialID(other.materialID) {}

    Triangle& operator=(const Triangle& other) {
        if (this != &other) {
            p1         = other.p1;
            p2         = other.p2;
            p3         = other.p3;
            n1         = other.n1;
            n2         = other.n2;
            n3         = other.n3;
            materialID = other.materialID;
        }
        return *this;
    }

    Triangle(Triangle&& other) noexcept
        : p1(std::move(other.p1)), p2(std::move(other.p2)), p3(std::move(other.p3)), n1(std::move(other.n1)), n2(std::move(other.n2)),
          n3(std::move(other.n3)), materialID(other.materialID) {}

    Triangle& operator=(Triangle&& other) noexcept {
        if (this != &other) {
            p1         = std::move(other.p1);
            p2         = std::move(other.p2);
            p3         = std::move(other.p3);
            n1         = std::move(other.n1);
            n2         = std::move(other.n2);
            n3         = std::move(other.n3);
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