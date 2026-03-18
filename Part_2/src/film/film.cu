#include "film/film.h"
#include "cuda/check.h"

#include <vector>

namespace pt {

static void createSurface(int w, int h, cudaArray_t& arr, cudaSurfaceObject_t& surf) {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&arr, &desc, w, h, cudaArraySurfaceLoadStore));

    cudaResourceDesc resDesc{};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = arr;
    CUDA_CHECK(cudaCreateSurfaceObject(&surf, &resDesc));
}

static void destroySurface(cudaArray_t& arr, cudaSurfaceObject_t& surf) {
    if (surf) { CUDA_CHECK(cudaDestroySurfaceObject(surf)); surf = 0; }
    if (arr)  { CUDA_CHECK(cudaFreeArray(arr));              arr = nullptr; }
}

void Film::init(int w, int h) {
    destroy();
    width_  = w;
    height_ = h;
    createSurface(w, h, arrayNew_, surfNew_);
    createSurface(w, h, arrayAcc_, surfAcc_);
}

void Film::destroy() {
    destroySurface(arrayNew_, surfNew_);
    destroySurface(arrayAcc_, surfAcc_);
    width_ = height_ = 0;
}

Film::~Film() {
    destroy();
}

void Film::clear() {
    if (!arrayAcc_) return;
    size_t rowBytes = static_cast<size_t>(width_) * sizeof(float4);
    std::vector<char> zeros(rowBytes * height_, 0);
    CUDA_CHECK(cudaMemcpy2DToArray(arrayAcc_, 0, 0, zeros.data(),
                                   rowBytes, rowBytes, height_,
                                   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2DToArray(arrayNew_, 0, 0, zeros.data(),
                                   rowBytes, rowBytes, height_,
                                   cudaMemcpyHostToDevice));
}

void Film::downloadAccumulated(float* hostDst) const {
    CUDA_CHECK(cudaMemcpy2DFromArray(
        hostDst, width_ * sizeof(float4),
        arrayAcc_, 0, 0,
        width_ * sizeof(float4), height_,
        cudaMemcpyDeviceToHost));
}

} // namespace pt
