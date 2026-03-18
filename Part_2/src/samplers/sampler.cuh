#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "samplers/pcg.cuh"
#include "samplers/sobol.cuh"

namespace pt {

struct Sampler {
    int          type;
    unsigned int pcgState;
    SobolSampler sobol;

    PT_D void initPCG(int samplerType, uint32_t frameCount, uint32_t sampleIdx) {
        type = samplerType;
        pcgState = rand_pcg_init_state(frameCount, sampleIdx);
    }

    PT_D void initSobol(int samplerType, int x, int y, uint32_t globalSampleIdx) {
        type = samplerType;
        sobol.start(x, y, globalSampleIdx);
    }

    PT_D float get1D() {
        if (type == 1)
            return sobol.get1D();
        return rand_pcg(&pcgState);
    }

    PT_D vec2 get2D() {
        float u = get1D();
        float v = get1D();
        return vec2(u, v);
    }
};

} // namespace pt
