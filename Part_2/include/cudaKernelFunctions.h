#pragma once

#include <cuda_runtime.h>
#include "SceneData.h"

// 计算核心
extern "C" void launchKernel(cudaSurfaceObject_t surface, pt::SceneData data);

// 混合
extern "C" void launchMixKernel(cudaSurfaceObject_t surfaceNew, cudaSurfaceObject_t surfaceAcc, pt::SceneData data);