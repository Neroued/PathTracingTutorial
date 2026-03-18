#pragma once

#include "cuda/check.h"
#include <cuda_runtime.h>

namespace pt {

struct Image;

class CudaTexture {
    cudaArray_t         cuArray_   = nullptr;
    cudaTextureObject_t cuTexture_ = 0;

public:
    CudaTexture() = default;
    explicit CudaTexture(const Image& img);
    ~CudaTexture();

    CudaTexture(const CudaTexture&)            = delete;
    CudaTexture& operator=(const CudaTexture&) = delete;

    CudaTexture(CudaTexture&& o) noexcept;
    CudaTexture& operator=(CudaTexture&& o) noexcept;

    cudaTextureObject_t handle() const { return cuTexture_; }
    explicit operator bool() const { return cuTexture_ != 0; }
};

} // namespace pt
