#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/math_utils.h"

namespace pt {

struct SurfaceInteraction {
    bool     hit;
    float    distance;
    int      materialId;
    int      primitiveId;
    vec3     point;
    vec3     normal;
    vec2     uv;
    float    baryU, baryV;

    PT_HD SurfaceInteraction()
        : hit(false), distance(float_max()), materialId(-1), primitiveId(-1),
          point(), normal(), uv(), baryU(0.0f), baryV(0.0f) {}
};

} // namespace pt
