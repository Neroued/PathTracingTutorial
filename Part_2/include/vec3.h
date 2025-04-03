#pragma once

#include "config.h"
#include "vector_functions.h"
#include <cuda_runtime.h>
#include "math_utils.h"

BEGIN_NAMESPACE_PT

// 使用 alignas(16) 保证 vec3 内存按 16 字节对齐，尽管三个 float 实际占 12 字节，但这种对齐可以提升 SIMD 以及 GPU 内存访问效率
struct alignas(16) vec3 {
    float x, y, z;

    // 默认构造函数：将向量初始化为 (0, 0, 0)
    PT_CPU_GPU vec3() : x(0.f), y(0.f), z(0.f) {}

    // 带参数构造函数
    PT_CPU_GPU vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    // 使用 float3 构造
    PT_CPU_GPU vec3(const float3& f) : x(f.x), y(f.y), z(f.z) {}

    // 默认复制构造函数和赋值运算符
    vec3(const vec3& v)            = default;
    vec3& operator=(const vec3& v) = default;

    // 加法运算符：向量逐元素相加
    PT_CPU_GPU vec3 operator+(const vec3& b) const { return vec3(x + b.x, y + b.y, z + b.z); }

    // 减法运算符：向量逐元素相减
    PT_CPU_GPU vec3 operator-(const vec3& b) const { return vec3(x - b.x, y - b.y, z - b.z); }

    // 标量乘法：向量的每个分量乘以标量 s
    PT_CPU_GPU vec3 operator*(float s) const { return vec3(x * s, y * s, z * s); }

    // 分量乘法
    PT_CPU_GPU vec3 operator*(const vec3& other) const { return vec3(x * other.x, y * other.y, z * other.z); }

    // 标量除法：向量的每个分量除以标量 s
    PT_CPU_GPU vec3 operator/(float s) const { return vec3(x / s, y / s, z / s); }

    // 加法赋值运算符
    PT_CPU_GPU vec3& operator+=(const vec3& b) {
        x += b.x;
        y += b.y;
        z += b.z;
        return *this;
    }

    // 减法赋值运算符
    PT_CPU_GPU vec3& operator-=(const vec3& b) {
        x -= b.x;
        y -= b.y;
        z -= b.z;
        return *this;
    }

    // 标量乘法赋值运算符
    PT_CPU_GPU vec3& operator*=(float s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    // 分量乘法赋值运算符
    PT_CPU_GPU vec3& operator*=(const vec3& other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }

    // 标量除法赋值运算符
    PT_CPU_GPU vec3& operator/=(float s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    // 一元负号运算符，实现向量取反，即各分量取相反数
    PT_CPU_GPU vec3 operator-() const { return vec3(-x, -y, -z); }

    // 下标运算符：允许通过索引访问向量的分量（非 const 版本）
    PT_CPU_GPU float& operator[](int i) { return *(&x + i); }

    // 下标运算符：允许通过索引访问向量的分量（const 版本）
    PT_CPU_GPU const float& operator[](int i) const { return *(&x + i); }

    // 点乘运算，数学公式为：
    //   dot(v, b) = x*b.x + y*b.y + z*b.z
    PT_CPU_GPU float dot(const vec3& b) const { return x * b.x + y * b.y + z * b.z; }

    // 叉乘运算，数学公式为：
    //   cross(v, b) = (y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x)
    PT_CPU_GPU vec3 cross(const vec3& b) const { return vec3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }

    // 计算向量的长度（欧几里得范数）：|v| = sqrt(x^2 + y^2 + z^2)
    PT_CPU_GPU float length() const {
        float sum = x * x + y * y + z * z;
        return pt::sqrt(sum);
    }

    // 归一化操作：返回一个与原向量同方向但长度为 1 的单位向量
    PT_CPU_GPU vec3 normalize() const {
        float len = length();
        // 防止除以零
        return (len > 0 ? (*this) / len : vec3(0.f, 0.f, 0.f));
    }

    // 重载相等运算符，用于判断两个向量是否相等
    PT_CPU_GPU bool operator==(const vec3& b) const { return x == b.x && y == b.y && z == b.z; }

    // 重载不等运算符
    PT_CPU_GPU bool operator!=(const vec3& b) const { return !(*this == b); }
};

// 非成员函数重载，实现标量与向量的乘法（标量在左侧）
inline PT_CPU_GPU vec3 operator*(float s, const vec3& v) { return v * s; }

inline PT_CPU_GPU vec3 cross(const vec3& a, const vec3& b) { return a.cross(b); }

inline PT_CPU_GPU float dot(const vec3& a, const vec3& b) { return a.dot(b); }

inline PT_CPU_GPU vec3 normalize(const vec3& v) { return v.normalize(); }

inline PT_CPU_GPU vec3 reflect(const vec3& I, const vec3& N) {
    float dotNI = I.x * N.x + I.y * N.y + I.z * N.z;
    float scale = 2.0f * dotNI;
    return vec3(I.x - scale * N.x, I.y - scale * N.y, I.z - scale * N.z);
}

END_NAMESPACE_PT

// 与 float3, float4 的转化
inline PT_CPU_GPU float3 make_float3(pt::vec3& v) { return make_float3(v.x, v.y, v.z); }

inline PT_CPU_GPU float4 make_float4(pt::vec3& v, float w) { return make_float4(v.x, v.y, v.z, w); }

inline PT_CPU_GPU float4 make_float4(float w, pt::vec3& v) { return make_float4(v.x, v.y, v.z, w); }
