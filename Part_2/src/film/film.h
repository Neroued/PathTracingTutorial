#pragma once

#include "pt.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace pt {

class Film {
public:
    Film() = default;
    ~Film();

    Film(const Film&)            = delete;
    Film& operator=(const Film&) = delete;

    void init(int w, int h);
    void destroy();

    cudaSurfaceObject_t newSurface()  const { return surfNew_; }
    cudaSurfaceObject_t accSurface()  const { return surfAcc_; }

    int width()  const { return width_; }
    int height() const { return height_; }

    void downloadAccumulated(float* hostDst) const;
    void clear();

    cudaArray_t accArray() const { return arrayAcc_; }

private:
    int width_  = 0;
    int height_ = 0;

    cudaArray_t         arrayNew_ = nullptr;
    cudaSurfaceObject_t surfNew_  = 0;

    cudaArray_t         arrayAcc_ = nullptr;
    cudaSurfaceObject_t surfAcc_  = 0;
};

} // namespace pt
