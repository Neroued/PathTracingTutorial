#pragma once

#include "pt.h"
#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#    include <cuda_runtime.h>
#endif

namespace pt {

PT_HD inline float min(float a, float b) {
#ifdef __CUDA_ARCH__
    return fminf(a, b);
#else
    return std::fmin(a, b);
#endif
}

PT_HD inline float max(float a, float b) {
#ifdef __CUDA_ARCH__
    return fmaxf(a, b);
#else
    return std::fmax(a, b);
#endif
}

PT_HD inline double min(double a, double b) {
#ifdef __CUDA_ARCH__
    return fmin(a, b);
#else
    return std::fmin(a, b);
#endif
}

PT_HD inline double max(double a, double b) {
#ifdef __CUDA_ARCH__
    return fmax(a, b);
#else
    return std::fmax(a, b);
#endif
}

PT_HD inline float clamp(float x, float lo, float hi) { return min(max(x, lo), hi); }

PT_HD inline float abs(float x) {
#ifdef __CUDA_ARCH__
    return fabsf(x);
#else
    return std::fabs(x);
#endif
}

PT_HD inline float sqrt(float x) {
#ifdef __CUDA_ARCH__
    return sqrtf(x);
#else
    return std::sqrt(x);
#endif
}

PT_HD inline float pow(float base, float exp) {
#ifdef __CUDA_ARCH__
    return powf(base, exp);
#else
    return std::pow(base, exp);
#endif
}

PT_HD inline float sin(float x) {
#ifdef __CUDA_ARCH__
    return sinf(x);
#else
    return std::sin(x);
#endif
}

PT_HD inline float cos(float x) {
#ifdef __CUDA_ARCH__
    return cosf(x);
#else
    return std::cos(x);
#endif
}

PT_HD inline float tan(float x) {
#ifdef __CUDA_ARCH__
    return tanf(x);
#else
    return std::tan(x);
#endif
}

PT_HD inline float asin(float x) {
#ifdef __CUDA_ARCH__
    return asinf(x);
#else
    return std::asin(x);
#endif
}

PT_HD inline float acos(float x) {
#ifdef __CUDA_ARCH__
    return acosf(x);
#else
    return std::acos(x);
#endif
}

PT_HD inline float atan(float x) {
#ifdef __CUDA_ARCH__
    return atanf(x);
#else
    return std::atan(x);
#endif
}

PT_HD inline float atan2(float y, float x) {
#ifdef __CUDA_ARCH__
    return atan2f(y, x);
#else
    return std::atan2(y, x);
#endif
}

PT_HD inline float exp(float x) {
#ifdef __CUDA_ARCH__
    return expf(x);
#else
    return std::exp(x);
#endif
}

PT_HD inline float log(float x) {
#ifdef __CUDA_ARCH__
    return logf(x);
#else
    return std::log(x);
#endif
}

PT_HD inline float floor(float x) {
#ifdef __CUDA_ARCH__
    return floorf(x);
#else
    return std::floor(x);
#endif
}

PT_HD inline float ceil(float x) {
#ifdef __CUDA_ARCH__
    return ceilf(x);
#else
    return std::ceil(x);
#endif
}

PT_HD inline float round(float x) {
#ifdef __CUDA_ARCH__
    return roundf(x);
#else
    return std::round(x);
#endif
}

PT_HD inline constexpr float float_max() { return FLT_MAX; }
PT_HD inline constexpr float float_min() { return -FLT_MAX; }
PT_HD inline constexpr float float_epsilon() { return FLT_EPSILON; }

} // namespace pt
