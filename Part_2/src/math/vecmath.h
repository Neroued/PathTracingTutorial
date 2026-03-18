#pragma once

#include "pt.h"
#include "math/math_utils.h"

#ifdef __CUDACC__
#    include <cuda_runtime.h>
#    include <vector_functions.h>
#endif

namespace pt {

// ---------------------------------------------------------------------------
// vec2
// ---------------------------------------------------------------------------
struct alignas(8) vec2 {
    float x, y;

    PT_HD vec2() : x(0.f), y(0.f) {}
    PT_HD vec2(float _x, float _y) : x(_x), y(_y) {}
#ifdef __CUDACC__
    PT_HD vec2(const float2& f) : x(f.x), y(f.y) {}
#endif
    vec2(const vec2&)            = default;
    vec2& operator=(const vec2&) = default;

    PT_HD vec2 operator+(const vec2& b) const { return {x + b.x, y + b.y}; }
    PT_HD vec2 operator-(const vec2& b) const { return {x - b.x, y - b.y}; }
    PT_HD vec2 operator*(float s) const { return {x * s, y * s}; }
    PT_HD vec2 operator*(const vec2& b) const { return {x * b.x, y * b.y}; }
    PT_HD vec2 operator/(float s) const { return {x / s, y / s}; }

    PT_HD vec2& operator+=(const vec2& b) { x += b.x; y += b.y; return *this; }
    PT_HD vec2& operator-=(const vec2& b) { x -= b.x; y -= b.y; return *this; }
    PT_HD vec2& operator*=(float s) { x *= s; y *= s; return *this; }
    PT_HD vec2& operator*=(const vec2& b) { x *= b.x; y *= b.y; return *this; }
    PT_HD vec2& operator/=(float s) { x /= s; y /= s; return *this; }

    PT_HD vec2 operator-() const { return {-x, -y}; }
    PT_HD float& operator[](int i) { return *(&x + i); }
    PT_HD const float& operator[](int i) const { return *(&x + i); }

    PT_HD float dot(const vec2& b) const { return x * b.x + y * b.y; }
    PT_HD float length() const { return pt::sqrt(x * x + y * y); }
    PT_HD vec2 normalize() const { float l = length(); return l > 0 ? *this / l : vec2(); }

    PT_HD bool operator==(const vec2& b) const { return x == b.x && y == b.y; }
    PT_HD bool operator!=(const vec2& b) const { return !(*this == b); }
};

inline PT_HD vec2 operator*(float s, const vec2& v) { return v * s; }
inline PT_HD float dot(const vec2& a, const vec2& b) { return a.dot(b); }
inline PT_HD vec2 normalize(const vec2& v) { return v.normalize(); }
inline PT_HD float cross(const vec2& a, const vec2& b) { return a.x * b.y - a.y * b.x; }

#ifdef __CUDACC__
inline PT_HD float2 make_float2(const vec2& v) { return ::make_float2(v.x, v.y); }
#endif

// ---------------------------------------------------------------------------
// vec3
// ---------------------------------------------------------------------------
struct alignas(16) vec3 {
    float x, y, z;

    PT_HD vec3() : x(0.f), y(0.f), z(0.f) {}
    PT_HD vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    PT_HD explicit vec3(float s) : x(s), y(s), z(s) {}
#ifdef __CUDACC__
    PT_HD vec3(const float3& f) : x(f.x), y(f.y), z(f.z) {}
#endif
    vec3(const vec3&)            = default;
    vec3& operator=(const vec3&) = default;

    PT_HD vec3 operator+(const vec3& b) const { return {x + b.x, y + b.y, z + b.z}; }
    PT_HD vec3 operator-(const vec3& b) const { return {x - b.x, y - b.y, z - b.z}; }
    PT_HD vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    PT_HD vec3 operator*(const vec3& b) const { return {x * b.x, y * b.y, z * b.z}; }
    PT_HD vec3 operator/(float s) const { return {x / s, y / s, z / s}; }

    PT_HD vec3& operator+=(const vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    PT_HD vec3& operator-=(const vec3& b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
    PT_HD vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    PT_HD vec3& operator*=(const vec3& b) { x *= b.x; y *= b.y; z *= b.z; return *this; }
    PT_HD vec3& operator/=(float s) { x /= s; y /= s; z /= s; return *this; }

    PT_HD vec3 operator-() const { return {-x, -y, -z}; }
    PT_HD float& operator[](int i) { return *(&x + i); }
    PT_HD const float& operator[](int i) const { return *(&x + i); }

    PT_HD float dot(const vec3& b) const { return x * b.x + y * b.y + z * b.z; }
    PT_HD vec3 cross(const vec3& b) const {
        return {y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x};
    }
    PT_HD float length() const { return pt::sqrt(x * x + y * y + z * z); }
    PT_HD vec3 normalize() const { float l = length(); return l > 0 ? *this / l : vec3(); }

    PT_HD bool operator==(const vec3& b) const { return x == b.x && y == b.y && z == b.z; }
    PT_HD bool operator!=(const vec3& b) const { return !(*this == b); }
};

inline PT_HD vec3 operator*(float s, const vec3& v) { return v * s; }
inline PT_HD vec3 cross(const vec3& a, const vec3& b) { return a.cross(b); }
inline PT_HD float dot(const vec3& a, const vec3& b) { return a.dot(b); }
inline PT_HD vec3 normalize(const vec3& v) { return v.normalize(); }

inline PT_HD vec3 reflect(const vec3& I, const vec3& N) {
    return I - 2.0f * dot(I, N) * N;
}

#ifdef __CUDACC__
inline PT_HD float3 make_float3(const vec3& v) { return ::make_float3(v.x, v.y, v.z); }
inline PT_HD float4 make_float4(const vec3& v, float w) { return ::make_float4(v.x, v.y, v.z, w); }
#endif

// ---------------------------------------------------------------------------
// vec4
// ---------------------------------------------------------------------------
struct alignas(16) vec4 {
    float x, y, z, w;

    PT_HD vec4() : x(0.f), y(0.f), z(0.f), w(0.f) {}
    PT_HD vec4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
    vec4(const vec4&)            = default;
    vec4& operator=(const vec4&) = default;
};

// ---------------------------------------------------------------------------
// Semantic type aliases
// ---------------------------------------------------------------------------
using Point3f  = vec3;
using Vector3f = vec3;
using Normal3f = vec3;
using Color3f  = vec3;
using Spectrum = Color3f;

} // namespace pt
