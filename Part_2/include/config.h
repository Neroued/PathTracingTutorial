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