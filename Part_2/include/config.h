#pragma once

#define BEGIN_NAMESPACE_PT namespace pt {
#define END_NAMESPACE_PT   }

#define PT_CPU __host__
#define PT_CPU_GPU __host__ __device__
#define PT_KERNEL __global__
#define PT_GPU __device__
