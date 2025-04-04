#pragma once

#include "config.h"
#include "vector_functions.h"
#include <cuda_runtime.h>
#include "math_utils.h"

BEGIN_NAMESPACE_PT

struct alignas(8) vec2 {
    float x, y;

    // 默认构造函数：将向量初始化为 (0, 0)
    PT_CPU_GPU vec2() : x(0.f), y(0.f) {}

    // 带参数构造函数
    PT_CPU_GPU vec2(float _x, float _y) : x(_x), y(_y) {}

    // 使用 float2 构造
    PT_CPU_GPU vec2(const float2& f) : x(f.x), y(f.y) {}

    // 默认复制构造函数和赋值运算符
    vec2(const vec2& v)            = default;
    vec2& operator=(const vec2& v) = default;

    // 加法运算符：向量逐元素相加
    PT_CPU_GPU vec2 operator+(const vec2& b) const { return vec2(x + b.x, y + b.y); }

    // 减法运算符：向量逐元素相减
    PT_CPU_GPU vec2 operator-(const vec2& b) const { return vec2(x - b.x, y - b.y); }

    // 标量乘法：向量的每个分量乘以标量 s
    PT_CPU_GPU vec2 operator*(float s) const { return vec2(x * s, y * s); }

    // 分量乘法：向量逐元素相乘
    PT_CPU_GPU vec2 operator*(const vec2& other) const { return vec2(x * other.x, y * other.y); }

    // 标量除法：向量的每个分量除以标量 s
    PT_CPU_GPU vec2 operator/(float s) const { return vec2(x / s, y / s); }

    // 加法赋值运算符
    PT_CPU_GPU vec2& operator+=(const vec2& b) {
        x += b.x;
        y += b.y;
        return *this;
    }

    // 减法赋值运算符
    PT_CPU_GPU vec2& operator-=(const vec2& b) {
        x -= b.x;
        y -= b.y;
        return *this;
    }

    // 标量乘法赋值运算符
    PT_CPU_GPU vec2& operator*=(float s) {
        x *= s;
        y *= s;
        return *this;
    }

    // 分量乘法赋值运算符
    PT_CPU_GPU vec2& operator*=(const vec2& other) {
        x *= other.x;
        y *= other.y;
        return *this;
    }

    // 标量除法赋值运算符
    PT_CPU_GPU vec2& operator/=(float s) {
        x /= s;
        y /= s;
        return *this;
    }

    // 一元负号运算符：实现向量取反（各分量取相反数）
    PT_CPU_GPU vec2 operator-() const { return vec2(-x, -y); }

    // 下标运算符：允许通过索引访问向量的分量（非 const 版本）
    PT_CPU_GPU float& operator[](int i) { return *(&x + i); }

    // 下标运算符：允许通过索引访问向量的分量（const 版本）
    PT_CPU_GPU const float& operator[](int i) const { return *(&x + i); }

    // 点乘运算，数学公式为：
    //   dot(a, b) = a.x * b.x + a.y * b.y
    PT_CPU_GPU float dot(const vec2& b) const { return x * b.x + y * b.y; }

    // 计算向量的长度（欧几里得范数），数学公式为：
    //   |v| = sqrt(x^2 + y^2)
    PT_CPU_GPU float length() const {
        float sum = x * x + y * y;
        return pt::sqrt(sum);
    }

    // 归一化操作：返回一个与原向量同方向但长度为 1 的单位向量
    PT_CPU_GPU vec2 normalize() const {
        float len = length();
        return (len > 0 ? (*this) / len : vec2(0.f, 0.f));
    }

    // 重载相等运算符，用于判断两个向量是否相等
    PT_CPU_GPU bool operator==(const vec2& b) const { return x == b.x && y == b.y; }

    // 重载不等运算符
    PT_CPU_GPU bool operator!=(const vec2& b) const { return !(*this == b); }
};

// 非成员函数重载

// 标量与向量的乘法（标量在左侧），数学公式为：
//   s * v = (s*v.x, s*v.y)
inline PT_CPU_GPU pt::vec2 operator*(float s, const pt::vec2& v) { return v * s; }

// 点乘函数，调用成员函数 dot
inline PT_CPU_GPU float dot(const pt::vec2& a, const pt::vec2& b) { return a.dot(b); }

// 归一化函数，调用成员函数 normalize
inline PT_CPU_GPU pt::vec2 normalize(const pt::vec2& v) { return v.normalize(); }

// 2D 叉乘运算（返回标量），数学公式为：
//   cross(a, b) = a.x * b.y - a.y * b.x
// 此值在二维中等价于扩展到三维时 z 分量的大小（方向由右手定则确定）
inline PT_CPU_GPU float cross(const pt::vec2& a, const pt::vec2& b) { return a.x * b.y - a.y * b.x; }

END_NAMESPACE_PT

// 与 float2 的转化
inline PT_CPU_GPU float2 make_float2(pt::vec2& v) { return make_float2(v.x, v.y); }