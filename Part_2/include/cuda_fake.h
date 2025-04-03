#pragma once
#ifdef __CLANGD__
#    include <__clang_cuda_builtin_vars.h>

// fake 声明 CUDA 的 surface 类型
template <typename T, int dim = 2>
struct surface {};

#    ifndef __device__
#        define __device__ __attribute__((device))
#    endif

// CUDA 内建函数伪声明
template <typename T, typename U>
__device__ void surf2Dwrite(T value, U surf, int x, int y);

template <typename T, typename U>
__device__ void surf2Dread(T* out, U surf, int x, int y);

#endif
