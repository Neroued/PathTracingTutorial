#include "cuda/cuda_texture.h"
#include "io/image_io.h"
#include <cstring>

namespace pt {

CudaTexture::~CudaTexture() {
    if (cuTexture_) CUDA_CHECK(cudaDestroyTextureObject(cuTexture_));
    if (cuArray_)   CUDA_CHECK(cudaFreeArray(cuArray_));
}

CudaTexture::CudaTexture(CudaTexture&& o) noexcept
    : cuArray_(o.cuArray_), cuTexture_(o.cuTexture_) {
    o.cuArray_   = nullptr;
    o.cuTexture_ = 0;
}

CudaTexture& CudaTexture::operator=(CudaTexture&& o) noexcept {
    if (this != &o) {
        if (cuTexture_) CUDA_CHECK(cudaDestroyTextureObject(cuTexture_));
        if (cuArray_)   CUDA_CHECK(cudaFreeArray(cuArray_));
        cuArray_     = o.cuArray_;
        cuTexture_   = o.cuTexture_;
        o.cuArray_   = nullptr;
        o.cuTexture_ = 0;
    }
    return *this;
}

static cudaTextureAddressMode glWrapToCuda(int glWrap) {
    switch (glWrap) {
    case 33071:  return cudaAddressModeClamp;   // GL_CLAMP_TO_EDGE
    case 33648:  return cudaAddressModeMirror;  // GL_MIRRORED_REPEAT
    case 10497:                                 // GL_REPEAT (default)
    default:     return cudaAddressModeWrap;
    }
}

CudaTexture::CudaTexture(const Image& img) {
    if (img.channel != 4) return;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&cuArray_, &channelDesc, img.width, img.height));

    size_t pitch = img.width * img.channel * sizeof(float);
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray_, 0, 0, img.data.data(), pitch, pitch, img.height,
                                   cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray_;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = glWrapToCuda(img.wrapS);
    texDesc.addressMode[1]   = glWrapToCuda(img.wrapT);
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&cuTexture_, &resDesc, &texDesc, nullptr));
}

} // namespace pt
