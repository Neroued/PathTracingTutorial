#pragma once

#include "pt.h"
#include "math/vecmath.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace pt {

// ---- Bit manipulation ----

PT_D inline uint32_t reverseBits32(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __brev(x);
#else
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
    return (x >> 16) | (x << 16);
#endif
}

// ---- Sobol base sequences (padded 2-base approach, Burley 2020) ----

// Dimension 0: Van der Corput (bit reversal of index)
PT_D inline uint32_t sobolDim0(uint32_t index) {
    return reverseBits32(index);
}

// Dimension 1: Sobol sequence with primitive polynomial x+1
// Direction numbers computed on-the-fly: v_i = v_{i-1} ^ (v_{i-1} >> 1)
PT_D inline uint32_t sobolDim1(uint32_t index) {
    uint32_t v = 0;
    uint32_t d = 0x80000000u;
    while (index != 0) {
        if (index & 1u) v ^= d;
        d ^= (d >> 1u);
        index >>= 1u;
    }
    return v;
}

// ---- Hash functions for Owen scrambling (Laine-Karras 2019) ----

PT_D inline uint32_t hashMix(uint32_t x) {
    x ^= x >> 17;
    x ^= x * 0xed5ad4bbu;
    x ^= x >> 11;
    x ^= x * 0xac4c1b51u;
    x ^= x >> 15;
    x ^= x * 0x31848babu;
    x ^= x >> 14;
    return x;
}

PT_D inline uint32_t owenHash(uint32_t x, uint32_t seed) {
    x += seed;
    x ^= x * 0x6c50b47cu;
    x ^= x * 0xb82f1e52u;
    x ^= x * 0xc7afe638u;
    x ^= x * 0x8d22f6e6u;
    return x;
}

// Full Owen scramble: reverse bits -> hash -> reverse back
PT_D inline uint32_t owenScramble(uint32_t v, uint32_t seed) {
    v = reverseBits32(v);
    v = owenHash(v, seed);
    v = reverseBits32(v);
    return v;
}

// Per-pixel, per-dimension seed for decorrelation
PT_D inline uint32_t dimensionSeed(int px, int py, uint32_t dim) {
    uint32_t h = static_cast<uint32_t>(px);
    h = hashMix(h + 0x9e3779b9u);
    h ^= static_cast<uint32_t>(py) + 0x9e3779b9u + (h << 6) + (h >> 2);
    h = hashMix(h);
    h ^= dim + 0x9e3779b9u + (h << 6) + (h >> 2);
    h = hashMix(h);
    return h;
}

// ---- Sobol Sampler (PBRT v4 PaddedSobolSampler approach) ----
//
// Key design: each get1D()/get2D() call generates a per-dimension shuffled
// sample index via Owen scramble. This ensures different sampling dimensions
// see independent parts of the Sobol sequence, eliminating inter-dimension
// correlation that plagued the previous global-shuffle approach.

struct SobolSampler {
    uint32_t sampleIndex;
    uint32_t dimension;
    int      px;
    int      py;

    PT_D void start(int x, int y, uint32_t globalSampleIdx) {
        px = x;
        py = y;
        sampleIndex = globalSampleIdx;
        dimension = 0;
    }

    PT_D float get1D() {
        uint32_t hash = dimensionSeed(px, py, dimension);
        uint32_t shuffledIdx = owenScramble(sampleIndex, hash);

        uint32_t raw = sobolDim0(shuffledIdx);
        uint32_t scrambled = owenScramble(raw, hashMix(hash));

        dimension++;
        constexpr float scale = 1.0f / 4294967296.0f;
        return pt::min(static_cast<float>(scrambled) * scale, 0.99999994f);
    }

    PT_D vec2 get2D() {
        uint32_t hash = dimensionSeed(px, py, dimension);
        uint32_t shuffledIdx = owenScramble(sampleIndex, hash);

        uint32_t raw0 = sobolDim0(shuffledIdx);
        uint32_t raw1 = sobolDim1(shuffledIdx);

        uint32_t seed0 = hashMix(hash);
        uint32_t seed1 = hashMix(seed0);
        uint32_t scrambled0 = owenScramble(raw0, seed0);
        uint32_t scrambled1 = owenScramble(raw1, seed1);

        dimension += 2;
        constexpr float scale = 1.0f / 4294967296.0f;
        return vec2(pt::min(static_cast<float>(scrambled0) * scale, 0.99999994f),
                    pt::min(static_cast<float>(scrambled1) * scale, 0.99999994f));
    }
};

} // namespace pt
