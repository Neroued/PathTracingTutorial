#pragma once

#include "pt.h"
#include "math/vecmath.h"

namespace pt {

struct Ray {
    vec3 origin;
    vec3 direction;

    Ray() = default;
    PT_HD Ray(const vec3& o, const vec3& d) : origin(o), direction(normalize(d)) {}
    PT_HD vec3 at(float t) const { return origin + t * direction; }
};

} // namespace pt
