#pragma once

#include "pt.h"
#include "math/vecmath.h"

#include <cmath>
#include <algorithm>

namespace pt {

struct mat4 {
    float m[16];

    PT_HD mat4() {
        for (int i = 0; i < 16; ++i) m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    }
    PT_HD explicit mat4(float diagonal) {
        for (int i = 0; i < 16; ++i) m[i] = 0.0f;
        m[0] = m[5] = m[10] = m[15] = diagonal;
    }
    PT_HD mat4(float m00, float m01, float m02, float m03,
               float m10, float m11, float m12, float m13,
               float m20, float m21, float m22, float m23,
               float m30, float m31, float m32, float m33) {
        m[0]=m00; m[1]=m01; m[2]=m02; m[3]=m03;
        m[4]=m10; m[5]=m11; m[6]=m12; m[7]=m13;
        m[8]=m20; m[9]=m21; m[10]=m22; m[11]=m23;
        m[12]=m30; m[13]=m31; m[14]=m32; m[15]=m33;
    }
    PT_HD explicit mat4(const float v[16]) { for (int i = 0; i < 16; ++i) m[i] = v[i]; }

    PT_HD mat4(const mat4&)            = default;
    PT_HD mat4& operator=(const mat4&) = default;

    // Element access
    PT_HD float& operator()(int row, int col) { return m[row * 4 + col]; }
    PT_HD const float& operator()(int row, int col) const { return m[row * 4 + col]; }
    PT_HD float* data() { return m; }
    PT_HD const float* data() const { return m; }

    // Comparison
    PT_HD bool operator==(const mat4& rhs) const {
        for (int i = 0; i < 16; ++i) if (m[i] != rhs.m[i]) return false;
        return true;
    }
    PT_HD bool operator!=(const mat4& rhs) const { return !(*this == rhs); }

    // Arithmetic
    PT_HD mat4 operator+(const mat4& rhs) const { mat4 r; for (int i=0;i<16;++i) r.m[i]=m[i]+rhs.m[i]; return r; }
    PT_HD mat4 operator-(const mat4& rhs) const { mat4 r; for (int i=0;i<16;++i) r.m[i]=m[i]-rhs.m[i]; return r; }
    PT_HD mat4 operator*(const mat4& rhs) const {
        mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                float s = 0.f;
                for (int k = 0; k < 4; ++k) s += (*this)(i, k) * rhs(k, j);
                r(i, j) = s;
            }
        return r;
    }
    PT_HD mat4& operator+=(const mat4& rhs) { for (int i=0;i<16;++i) m[i]+=rhs.m[i]; return *this; }
    PT_HD mat4& operator-=(const mat4& rhs) { for (int i=0;i<16;++i) m[i]-=rhs.m[i]; return *this; }
    PT_HD mat4& operator*=(const mat4& rhs) { *this = *this * rhs; return *this; }

    PT_HD mat4 operator*(float s) const { mat4 r; for (int i=0;i<16;++i) r.m[i]=m[i]*s; return r; }
    PT_HD mat4 operator/(float s) const { float inv=1.f/s; return *this * inv; }
    PT_HD mat4& operator*=(float s) { for (int i=0;i<16;++i) m[i]*=s; return *this; }
    PT_HD mat4& operator/=(float s) { return *this *= (1.f/s); }
    PT_HD mat4 operator-() const { mat4 r; for (int i=0;i<16;++i) r.m[i]=-m[i]; return r; }

    PT_HD friend mat4 operator*(float s, const mat4& M) { return M * s; }

    // Transpose / Inverse
    PT_HD mat4 transposed() const {
        mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) r(j, i) = (*this)(i, j);
        return r;
    }

    PT_HD mat4 inverse() const {
        mat4 inv = identity();
        mat4 A = *this;
        for (int i = 0; i < 4; ++i) {
            float pivotVal = std::fabs(A(i, i));
            int pivotRow = i;
            for (int j = i + 1; j < 4; ++j) {
                float c = std::fabs(A(j, i));
                if (c > pivotVal) { pivotVal = c; pivotRow = j; }
            }
            if (pivotVal < 1e-8f) return identity();
            if (pivotRow != i) {
                for (int k = 0; k < 4; ++k) {
                    std::swap(A(i, k), A(pivotRow, k));
                    std::swap(inv(i, k), inv(pivotRow, k));
                }
            }
            float pivot = A(i, i);
            for (int k = 0; k < 4; ++k) { A(i,k)/=pivot; inv(i,k)/=pivot; }
            for (int j = 0; j < 4; ++j) {
                if (j != i) {
                    float f = A(j, i);
                    for (int k = 0; k < 4; ++k) { A(j,k)-=f*A(i,k); inv(j,k)-=f*inv(i,k); }
                }
            }
        }
        return inv;
    }

    // Transform helpers
    PT_HD vec3 transformPoint(const vec3& v) const {
        return {m[0]*v.x+m[1]*v.y+m[2]*v.z+m[3],
                m[4]*v.x+m[5]*v.y+m[6]*v.z+m[7],
                m[8]*v.x+m[9]*v.y+m[10]*v.z+m[11]};
    }
    PT_HD vec3 transformVector(const vec3& v) const {
        return {m[0]*v.x+m[1]*v.y+m[2]*v.z,
                m[4]*v.x+m[5]*v.y+m[6]*v.z,
                m[8]*v.x+m[9]*v.y+m[10]*v.z};
    }
    PT_HD vec4 operator*(const vec4& v) const {
        return {m[0]*v.x+m[1]*v.y+m[2]*v.z+m[3]*v.w,
                m[4]*v.x+m[5]*v.y+m[6]*v.z+m[7]*v.w,
                m[8]*v.x+m[9]*v.y+m[10]*v.z+m[11]*v.w,
                m[12]*v.x+m[13]*v.y+m[14]*v.z+m[15]*v.w};
    }
    PT_HD vec3 operator*(const vec3& v) const { return transformPoint(v); }

    PT_HD mat4& translate(const vec3& t) { *this = *this * translation(t); return *this; }
    PT_HD mat4& rotate(float angleDeg, const vec3& axis) {
        *this = *this * rotation(angleDeg * Pi / 180.0f, axis);
        return *this;
    }

    // Static factories
    PT_HD static mat4 identity() { return mat4(); }
    PT_HD static mat4 zero() { mat4 r; for (int i=0;i<16;++i) r.m[i]=0.f; return r; }

    PT_HD static mat4 translation(float tx, float ty, float tz) {
        mat4 r; r.m[3]=tx; r.m[7]=ty; r.m[11]=tz; return r;
    }
    PT_HD static mat4 translation(const vec3& t) { return translation(t.x, t.y, t.z); }

    PT_HD static mat4 scaling(float sx, float sy, float sz) {
        mat4 r; r.m[0]=sx; r.m[5]=sy; r.m[10]=sz; return r;
    }
    PT_HD static mat4 scaling(const vec3& s) { return scaling(s.x, s.y, s.z); }

    PT_HD static mat4 rotation(float rad, const vec3& axis) {
        return rotation(rad, axis.x, axis.y, axis.z);
    }
    PT_HD static mat4 rotation(float rad, float ax, float ay, float az) {
        float len = std::sqrt(ax*ax + ay*ay + az*az);
        if (len < 1e-8f) return identity();
        float x=ax/len, y=ay/len, z=az/len;
        float c = cosf(rad), s = sinf(rad), omc = 1.0f - c;
        mat4 r;
        r.m[0]=c+omc*x*x;     r.m[1]=omc*x*y-s*z;   r.m[2]=omc*x*z+s*y;   r.m[3]=0;
        r.m[4]=omc*y*x+s*z;   r.m[5]=c+omc*y*y;      r.m[6]=omc*y*z-s*x;   r.m[7]=0;
        r.m[8]=omc*z*x-s*y;   r.m[9]=omc*z*y+s*x;    r.m[10]=c+omc*z*z;    r.m[11]=0;
        r.m[12]=0; r.m[13]=0; r.m[14]=0; r.m[15]=1;
        return r;
    }
};

} // namespace pt
