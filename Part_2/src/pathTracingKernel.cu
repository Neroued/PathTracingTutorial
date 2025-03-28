#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <QDebug>
#include <surface_types.h>
#include <surface_functions.h>
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
    if (err != cudaSuccess) qFatal() << "CUDA Kernel failed: " << cudaGetErrorString(err);
}