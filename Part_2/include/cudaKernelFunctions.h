#pragma once

#include <cuda_runtime.h>

// 计算核心
extern "C" void launchKernel(cudaSurfaceObject_t surface, int width, int height);

// 混合
extern "C" void launchMixKernel(cudaSurfaceObject_t surfaceNew, cudaSurfaceObject_t surfaceAcc, int width, int height, unsigned int sampleCount);