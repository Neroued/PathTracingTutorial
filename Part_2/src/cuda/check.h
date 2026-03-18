#pragma once

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (cudaSuccess != err) {                                                                  \
            fprintf(stderr, "CUDA error in %s:%d : %s\n", __FILE__, __LINE__,                      \
                    cudaGetErrorString(err));                                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK_LAST()                                                                          \
    do {                                                                                           \
        cudaError_t err = cudaGetLastError();                                                      \
        if (cudaSuccess != err) {                                                                  \
            fprintf(stderr, "CUDA error in %s:%d : %s\n", __FILE__, __LINE__,                      \
                    cudaGetErrorString(err));                                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
