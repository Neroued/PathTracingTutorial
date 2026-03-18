#pragma once

#include "pt.h"
#include "math/vecmath.h"

namespace pt {

enum class LightType : int {
    Point       = 0,
    Directional = 1,
    Spot        = 2,
};

struct PunctualLight {
    vec3      position    = {0,0,0};
    float     range       = Infinity;
    vec3      color       = {1,1,1};
    float     intensity   = 1.0f;
    vec3      direction   = {0,-1,0};
    float     innerCone   = 0.0f;        // cos(inner angle)
    float     outerCone   = 0.7071068f;  // cos(outer angle), default ~45deg
    LightType type        = LightType::Point;
    float     pad0_       = 0.0f;
    float     pad1_       = 0.0f;
};

PT_HD inline float spotAttenuation(const PunctualLight& light, const vec3& toLight) {
    float cosAngle = dot(normalize(-toLight), light.direction);
    if (cosAngle < light.outerCone) return 0.0f;
    if (cosAngle > light.innerCone) return 1.0f;
    float t = (cosAngle - light.outerCone) / (light.innerCone - light.outerCone + 1e-7f);
    return t * t;
}

PT_HD inline float distanceAttenuation(float dist, float range) {
    if (dist >= range) return 0.0f;
    return 1.0f / (dist * dist + 1e-7f);
}

} // namespace pt
