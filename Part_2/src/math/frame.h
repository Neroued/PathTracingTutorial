#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/math_utils.h"

namespace pt {

// Orthonormal basis (TBN) constructed from a normal vector.
// Used to convert between local shading coordinates and world coordinates.
struct Frame {
    vec3 tangent;
    vec3 bitangent;
    vec3 normal;

    Frame() = default;

    PT_HD explicit Frame(const vec3& n) : normal(n) {
        vec3 helper(1.0f, 0.0f, 0.0f);
        if (pt::abs(n.x) > 0.999f) helper = vec3(0.0f, 0.0f, 1.0f);
        tangent   = normalize(cross(n, helper));
        bitangent = normalize(cross(n, tangent));
    }

    // Transform a direction from local coordinates to world coordinates.
    PT_HD vec3 toWorld(const vec3& v) const {
        return v.x * tangent + v.y * bitangent + v.z * normal;
    }

    // Transform a direction from world coordinates to local (shading) coordinates.
    PT_HD vec3 toLocal(const vec3& v) const {
        return vec3(dot(v, tangent), dot(v, bitangent), dot(v, normal));
    }
};

} // namespace pt
