#pragma once

#include "config.h"
#include "ImageLoader.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "texture_types.h"
#include "vector_types.h"
#include "Texture.h"
#include <__msvc_chrono.hpp>
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>

BEGIN_NAMESPACE_PT

Texture::~Texture() {
    if (cuTexture) CUDA_SAFE_CALL(cudaDestroyTextureObject(cuTexture));
    if (cuArray) CUDA_SAFE_CALL(cudaFreeArray(cuArray));
}

Texture::Texture(Texture&& other) {
    cuArray         = other.cuArray;
    cuTexture       = other.cuTexture;
    other.cuArray   = nullptr;
    other.cuTexture = 0;
}

Texture& Texture::operator=(Texture&& other) {
    if (this != &other) {
        if (cuTexture) CUDA_SAFE_CALL(cudaDestroyTextureObject(cuTexture));
        if (cuArray) CUDA_SAFE_CALL(cudaFreeArray(cuArray));
        cuArray         = other.cuArray;
        cuTexture       = other.cuTexture;
        other.cuArray   = nullptr;
        other.cuTexture = 0;
    }
    return *this;
}

Texture::Texture(const Image& img) {
    // 要求 img 为 4 通道
    if (img.channel != 4) return;

    // 创建通道描述符
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    // 分配内存
    CUDA_SAFE_CALL(cudaMallocArray(&cuArray, &channelDesc, img.width, img.height));

    // 拷贝内存
    size_t pitch = img.width * img.channel * sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpy2DToArray(cuArray, 0, 0, img.data.data(), pitch, pitch, img.height, cudaMemcpyHostToDevice));

    // 定义资源描述符
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // 定义纹理描述符
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;    // 水平方向 clamp
    texDesc.addressMode[1]   = cudaAddressModeClamp;    // 竖直方向 clamp
    texDesc.filterMode       = cudaFilterModeLinear;    // 线性过滤
    texDesc.readMode         = cudaReadModeElementType; // 按数据类型读取
    texDesc.normalizedCoords = 1;                       // 归一化纹理坐标

    // 创建纹理对象
    CUDA_SAFE_CALL(cudaCreateTextureObject(&cuTexture, &resDesc, &texDesc, nullptr));
}

END_NAMESPACE_PT