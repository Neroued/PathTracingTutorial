#pragma once

#include "config.h"

#include "cuda_fake.h"
#include <cuda_runtime.h>

BEGIN_NAMESPACE_PT

// 生成 [0, 1) 之间随机数，需要输入一个全局的变量 state
PT_GPU inline float rand_pcg(unsigned int* state) {
    unsigned int oldstate = *state;
    *state                = oldstate * 747796405u + 2891336453u;
    unsigned int word     = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    return static_cast<float>((word >> 22u) ^ word) / 4294967296.0f;
}

// 使用线程 ID 与 frameCount 初始化一个种子
PT_GPU inline unsigned int rand_pcg_init_state(unsigned int frameCount) {
    int x              = blockIdx.x * blockDim.x + threadIdx.x;
    int y              = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int state = static_cast<unsigned int>(static_cast<unsigned int>(x) * 1973u + static_cast<unsigned int>(y) * 9277u +
                                                   static_cast<unsigned int>(frameCount) * 26699u) |
                         1u;
    return state;
}

END_NAMESPACE_PT