#pragma once

#include "pt.h"

#ifdef __CUDACC__
#    include <cuda_runtime.h>
#endif

namespace pt {

PT_D inline float rand_pcg(unsigned int* state) {
    unsigned int old = *state;
    *state           = old * 747796405u + 2891336453u;
    unsigned int word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return static_cast<float>((word >> 22u) ^ word) / 4294967296.0f;
}

PT_D inline unsigned int rand_pcg_init_state(unsigned int frameCount, unsigned int sampleIdx = 0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int seed = static_cast<unsigned int>(x) * 1973u +
                        static_cast<unsigned int>(y) * 9277u +
                        frameCount * 26699u +
                        sampleIdx * 12149u;
    seed = seed * 747796405u + 2891336453u;
    return seed | 1u;
}

} // namespace pt
