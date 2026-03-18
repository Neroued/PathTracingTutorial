#pragma once

#include <cstdint>
#include <cfloat>
#include <limits>

// CUDA qualifier macros -- degrade gracefully when compiled by a pure C++ compiler
#ifdef __CUDACC__
#    define PT_HD     __host__ __device__
#    define PT_D      __device__
#    define PT_H      __host__
#    define PT_GLOBAL __global__
#else
#    define PT_HD
#    define PT_D
#    define PT_H
#    define PT_GLOBAL
#endif

// Clangd compatibility shims for CUDA built-in types
#ifdef __CLANGD__
#    include <__clang_cuda_builtin_vars.h>
template <typename T, int dim = 2>
struct surface {};
#    ifndef __device__
#        define __device__ __attribute__((device))
#    endif
template <typename T, typename U>
__device__ void surf2Dwrite(T value, U surf, int x, int y);
template <typename T, typename U>
__device__ void surf2Dread(T* out, U surf, int x, int y);
#endif

namespace pt {

constexpr float Pi       = 3.14159265358979323846f;
constexpr float InvPi    = 0.31830988618379067154f;
constexpr float Inv2Pi   = 0.15915494309189533577f;
constexpr float Inv4Pi   = 0.07957747154594766788f;
constexpr float TwoPi    = 6.28318530717958647692f;
constexpr float Infinity = FLT_MAX;
constexpr float Epsilon  = FLT_EPSILON;

constexpr int BlockSizeX = 16;
constexpr int BlockSizeY = 16;

} // namespace pt
