#pragma once

#include "config.h"

#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#    include <cuda_runtime.h>
#endif

BEGIN_NAMESPACE_PT

// float min/max
PT_CPU_GPU inline float min(float a, float b) {
#ifdef __CUDA_ARCH__
    return fminf(a, b);
#else
    return std::fmin(a, b);
#endif
}

PT_CPU_GPU inline float max(float a, float b) {
#ifdef __CUDA_ARCH__
    return fmaxf(a, b);
#else
    return std::fmax(a, b);
#endif
}

// double min/max
PT_CPU_GPU inline double min(double a, double b) {
#ifdef __CUDA_ARCH__
    return fmin(a, b);
#else
    return std::fmin(a, b);
#endif
}

PT_CPU_GPU inline double max(double a, double b) {
#ifdef __CUDA_ARCH__
    return fmax(a, b);
#else
    return std::fmax(a, b);
#endif
}

// clamp for float
PT_CPU_GPU inline float clamp(float x, float a, float b) { return min(max(x, a), b); }

// abs
PT_CPU_GPU inline float abs(float x) {
#ifdef __CUDA_ARCH__
    return fabsf(x);
#else
    return std::fabs(x);
#endif
}

// sqrt
PT_CPU_GPU inline float sqrt(float x) {
#ifdef __CUDA_ARCH__
    return sqrtf(x);
#else
    return std::sqrt(x);
#endif
}

// pow
PT_CPU_GPU inline float pow(float base, float exp) {
#ifdef __CUDA_ARCH__
    return powf(base, exp);
#else
    return std::pow(base, exp);
#endif
}

// sin, cos
PT_CPU_GPU inline float sin(float x) {
#ifdef __CUDA_ARCH__
    return sinf(x);
#else
    return std::sin(x);
#endif
}

PT_CPU_GPU inline float cos(float x) {
#ifdef __CUDA_ARCH__
    return cosf(x);
#else
    return std::cos(x);
#endif
}

// exp, log
PT_CPU_GPU inline float exp(float x) {
#ifdef __CUDA_ARCH__
    return expf(x);
#else
    return std::exp(x);
#endif
}

PT_CPU_GPU inline float log(float x) {
#ifdef __CUDA_ARCH__
    return logf(x);
#else
    return std::log(x);
#endif
}

// floor, ceil, round
PT_CPU_GPU inline float floor(float x) {
#ifdef __CUDA_ARCH__
    return floorf(x);
#else
    return std::floor(x);
#endif
}

PT_CPU_GPU inline float ceil(float x) {
#ifdef __CUDA_ARCH__
    return ceilf(x);
#else
    return std::ceil(x);
#endif
}

PT_CPU_GPU inline float round(float x) {
#ifdef __CUDA_ARCH__
    return roundf(x);
#else
    return std::round(x);
#endif
}

PT_CPU_GPU inline constexpr float float_max() {
    return FLT_MAX;
}

PT_CPU_GPU inline constexpr float float_min() {
    return -FLT_MAX;
}

PT_CPU_GPU inline constexpr float float_epsilon() {
    return FLT_EPSILON;
}

END_NAMESPACE_PT

