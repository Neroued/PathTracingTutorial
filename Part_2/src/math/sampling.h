#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/math_utils.h"

namespace pt {

// Uniform hemisphere sampling -- returns direction in local frame where z = up
PT_D inline vec3 sampleUniformHemisphere(float u1, float u2) {
    float z   = u1;
    float r   = pt::max(0.0f, pt::sqrt(1.0f - z * z));
    float phi = TwoPi * u2;
    return {r * pt::cos(phi), r * pt::sin(phi), z};
}

// Cosine-weighted hemisphere sampling (Malley's method)
PT_D inline vec3 sampleCosineHemisphere(float u1, float u2) {
    float r     = pt::sqrt(u1);
    float phi   = TwoPi * u2;
    float x     = r * pt::cos(phi);
    float y     = r * pt::sin(phi);
    float z     = pt::sqrt(pt::max(0.0f, 1.0f - u1));
    return {x, y, z};
}

// Uniform disk sampling (concentric mapping)
PT_D inline vec2 sampleUniformDisk(float u1, float u2) {
    float r   = pt::sqrt(u1);
    float phi = TwoPi * u2;
    return {r * pt::cos(phi), r * pt::sin(phi)};
}

PT_HD inline float powerHeuristic(float pdfA, float pdfB) {
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    float sum = a2 + b2;
    return sum > 0.0f ? a2 / sum : 0.0f;
}

} // namespace pt
