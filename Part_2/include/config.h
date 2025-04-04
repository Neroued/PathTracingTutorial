#pragma once

#define BEGIN_NAMESPACE_PT namespace pt {
#define END_NAMESPACE_PT   }

#ifdef __CUDACC__
#    define PT_CPU     __host__
#    define PT_CPU_GPU __host__ __device__
#    define PT_KERNEL  __global__
#    define PT_GPU     __device__
#else
#    define PT_CPU
#    define PT_CPU_GPU
#    define PT_KERNEL
#    define PT_GPU
#endif

constexpr int BLOCK_SIZE_X = 16;
constexpr int BLOCK_SIZE_Y = 16;

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                                                                                                         \
    do {                                                                                                                                             \
        cudaError_t err = call;                                                                                                                      \
        if (cudaSuccess != err) {                                                                                                                    \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                                                                                                      \
        }                                                                                                                                            \
    } while (0)

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                                                                                                         \
    do {                                                                                                                                             \
        /* Check synchronous errors, i.e. pre-launch */                                                                                              \
        cudaError_t err = cudaGetLastError();                                                                                                        \
        if (cudaSuccess != err) {                                                                                                                    \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                                                                                                      \
        }                                                                                                                                            \
    } while (0)