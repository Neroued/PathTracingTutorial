#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <surface_functions.h>
#include <surface_types.h>
#include <vector_functions.h>
#include <vector_types.h>

__global__ void kernelTest(cudaSurfaceObject_t surface, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float normX = static_cast<float>(x) / static_cast<float>(width);
    float normY = static_cast<float>(y) / static_cast<float>(height);

    float4 value = make_float4(normX, normY, 0.0f, 1.0f);

    int byteOffset = x * sizeof(float4);
    surf2Dwrite(value, surface, byteOffset, y);
}

extern "C" void launchKernel(cudaSurfaceObject_t surface, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    kernelTest<<<gridSize, blockSize>>>(surface, width, height);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cerr << "CUDA Kernel failed: " << cudaGetErrorString(err) << std::endl;
}

__global__ void kernelMix(cudaSurfaceObject_t surfaceNew, cudaSurfaceObject_t surfaceAcc, int width, int height, unsigned int sampleCount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int byteOffset = x * sizeof(float4);

    float4 newSample, accSample;
    surf2Dread(&newSample, surfaceNew, byteOffset, y);
    surf2Dread(&accSample, surfaceAcc, byteOffset, y);

    float inv_sampleCount = 1.0f / sampleCount;

    accSample.x = accSample.x * (1.0f - inv_sampleCount) + newSample.x * inv_sampleCount;
    accSample.y = accSample.y * (1.0f - inv_sampleCount) + newSample.y * inv_sampleCount;
    accSample.z = accSample.z * (1.0f - inv_sampleCount) + newSample.z * inv_sampleCount;
    accSample.w = accSample.w * (1.0f - inv_sampleCount) + newSample.w * inv_sampleCount;

    surf2Dwrite(accSample, surfaceAcc, byteOffset, y);
}

extern "C" void launchMixKernel(cudaSurfaceObject_t surfaceNew, cudaSurfaceObject_t surfaceAcc, int width, int height, unsigned int sampleCount) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    kernelMix<<<gridSize, blockSize>>>(surfaceNew, surfaceAcc, width, height, sampleCount);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cerr << "CUDA Kernel failed: " << cudaGetErrorString(err) << std::endl;
}