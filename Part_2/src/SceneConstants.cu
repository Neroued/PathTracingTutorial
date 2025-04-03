#include "config.h"

#include "SceneConstants.cuh"
#include <cuda_runtime.h>
#include <cstdio>


BEGIN_NAMESPACE_PT

__constant__ SceneConstants d_sceneConstants;

PT_CPU void uploadSceneConstant(const SceneConstants& host) {
    cudaMemcpyToSymbol(d_sceneConstants, &host, sizeof(pt::SceneConstants));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err)); }
}

END_NAMESPACE_PT