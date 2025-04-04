#include "config.h"

#include "SceneConstants.cuh"
#include <cuda_runtime.h>
#include <cstdio>


BEGIN_NAMESPACE_PT

__constant__ SceneConstants d_sceneConstants;

PT_CPU void uploadSceneConstant(const SceneConstants& host) {
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_sceneConstants, &host, sizeof(pt::SceneConstants)));
}

END_NAMESPACE_PT