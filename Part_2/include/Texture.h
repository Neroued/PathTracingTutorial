#pragma once

#include "config.h"
#include <cuda_runtime.h>


BEGIN_NAMESPACE_PT

struct Image;

struct Texture {
    cudaArray_t cuArray;
    cudaTextureObject_t cuTexture;

    Texture() : cuArray(nullptr), cuTexture(0) {}

    // 从 Image 转换
    Texture(const Image& img);

    ~Texture();

    // 禁用拷贝构造/赋值
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    // 显式声明移动构造与赋值
    Texture(Texture&&);
    Texture& operator=(Texture&&);
};

END_NAMESPACE_PT