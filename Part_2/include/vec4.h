#pragma once

#include "config.h"
#include <cuda_runtime.h>

BEGIN_NAMESPACE_PT

// 使用 alignas(16) 保证 vec3 内存按 16 字节对齐，尽管三个 float 实际占 12 字节，但这种对齐可以提升 SIMD 以及 GPU 内存访问效率
struct alignas(16) vec4 {
    float x, y, z, w;
};

END_NAMESPACE_PT