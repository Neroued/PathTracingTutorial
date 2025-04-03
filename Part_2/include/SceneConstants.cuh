#pragma once

#include "config.h"

constexpr float PI     = 3.14159265358979323846;
constexpr float TWO_PI = 2.0 * PI;
constexpr float INV_PI = 1.0 / PI;

BEGIN_NAMESPACE_PT

struct SceneConstants {
    int width;
    int height;
    float3 cameraPos;
    float screenZ;
    int samplePerFrame;
    int targetSamples;
    int depth;
};

PT_CPU void uploadSceneConstant(const SceneConstants& host);

#ifdef __CUDACC__
extern __constant__ SceneConstants d_sceneConstants;
#endif

END_NAMESPACE_PT