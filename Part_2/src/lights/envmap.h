#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/math_utils.h"

#ifdef __CUDACC__
#    include <cuda_runtime.h>
#    include <texture_indirect_functions.h>
#endif

namespace pt {

// Convert a 3D direction to equirectangular UV coordinates
PT_HD inline vec2 sampleSphericalMap(const vec3& direction) {
    vec2 uv(pt::atan2(direction.z, direction.x), pt::asin(direction.y));
    uv.x *= Inv2Pi;
    uv.y *= InvPi;
    uv.x += 0.5f;
    uv.y += 0.5f;
    uv.y = 1.0f - uv.y;
    return uv;
}

PT_HD inline vec3 directionFromSphericalMap(const vec2& uv) {
    float phi      = (uv.x - 0.5f) * TwoPi;
    float theta    = uv.y * Pi;
    float sinTheta = pt::sin(theta);
    return normalize(vec3(pt::cos(phi) * sinTheta,
                          pt::cos(theta),
                          pt::sin(phi) * sinTheta));
}

#ifdef __CUDACC__
// Sample the HDR environment map texture
PT_D inline vec3 sampleHDR(cudaTextureObject_t texObj, const vec3& direction) {
    if (texObj == 0) return vec3(0.0f);
    vec2 uv = sampleSphericalMap(direction);
    float4 color = tex2D<float4>(texObj, uv.x, uv.y);
    return {color.x, color.y, color.z};
}

PT_D inline vec3 sampleEnvironmentRadiance(cudaTextureObject_t texObj, const vec3& direction) {
    return sampleHDR(texObj, direction) * TwoPi;
}
#endif

} // namespace pt
