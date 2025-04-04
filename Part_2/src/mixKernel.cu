#include "config.h"

#include <cuda_runtime.h>
#include "cuda_fake.h"
#include "SceneData.h"

PT_KERNEL void kernelMix(cudaSurfaceObject_t surfaceNew, cudaSurfaceObject_t surfaceAcc, pt::SceneData data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= data.width || y >= data.height) return;

    int byteOffset = x * sizeof(float4);

    float4 newSample, accSample;
    surf2Dread(&newSample, surfaceNew, byteOffset, y);
    surf2Dread(&accSample, surfaceAcc, byteOffset, y);

    float inv_sampleCount = 1.0f / (data.frameCount + 1);

    accSample.x = accSample.x * (1.0f - inv_sampleCount) + newSample.x * inv_sampleCount;
    accSample.y = accSample.y * (1.0f - inv_sampleCount) + newSample.y * inv_sampleCount;
    accSample.z = accSample.z * (1.0f - inv_sampleCount) + newSample.z * inv_sampleCount;
    accSample.w = accSample.w * (1.0f - inv_sampleCount) + newSample.w * inv_sampleCount;

    surf2Dwrite(accSample, surfaceAcc, byteOffset, y);
}

extern "C" void launchMixKernel(cudaSurfaceObject_t surfaceNew, cudaSurfaceObject_t surfaceAcc, pt::SceneData data) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((data.width + blockSize.x - 1) / blockSize.x, (data.height + blockSize.y - 1) / blockSize.y);
    kernelMix<<<gridSize, blockSize>>>(surfaceNew, surfaceAcc, data);

    CHECK_LAUNCH_ERROR();
}