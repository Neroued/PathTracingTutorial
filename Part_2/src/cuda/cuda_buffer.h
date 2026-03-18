#pragma once

#include "cuda/check.h"
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

namespace pt {

template <typename T>
class CudaBuffer {
    T*     d_ptr_ = nullptr;
    size_t size_  = 0;

public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) { resize(count); }

    CudaBuffer(const T* hostData, size_t count) { upload(hostData, count); }

    ~CudaBuffer() { free(); }

    CudaBuffer(CudaBuffer&& o) noexcept : d_ptr_(o.d_ptr_), size_(o.size_) {
        o.d_ptr_ = nullptr;
        o.size_  = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& o) noexcept {
        if (this != &o) {
            free();
            d_ptr_   = o.d_ptr_;
            size_    = o.size_;
            o.d_ptr_ = nullptr;
            o.size_  = 0;
        }
        return *this;
    }

    CudaBuffer(const CudaBuffer&)            = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    void upload(const T* hostData, size_t count) {
        if (count == 0) {
            resize(0);
            return;
        }
        if (size_ != count) resize(count);
        CUDA_CHECK(cudaMemcpy(d_ptr_, hostData, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void upload(const std::vector<T>& v) { upload(v.data(), v.size()); }

    void download(T* hostDst, size_t count) const {
        CUDA_CHECK(cudaMemcpy(hostDst, d_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void download(std::vector<T>& v) const {
        v.resize(size_);
        download(v.data(), size_);
    }

    void resize(size_t count) {
        if (count == size_ && d_ptr_) return;
        free();
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr_, count * sizeof(T)));
            size_ = count;
        }
    }

    void free() {
        if (d_ptr_) {
            CUDA_CHECK(cudaFree(d_ptr_));
            d_ptr_ = nullptr;
        }
        size_ = 0;
    }

    T*     data()  const { return d_ptr_; }
    size_t size()  const { return size_; }
    bool   empty() const { return size_ == 0; }
};

} // namespace pt
