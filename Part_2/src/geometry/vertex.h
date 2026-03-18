#pragma once

#include "pt.h"
#include "math/vecmath.h"

#include <cstdint>

namespace pt {

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 uv;
    vec4 tangent;   // xyz = tangent direction, w = handedness sign (+1/-1)
};

struct TriangleFace {
    uint32_t v0, v1, v2;
    int      materialID = 0;
};

} // namespace pt
