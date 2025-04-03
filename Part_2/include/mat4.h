#pragma once

#include <cmath> // for std::fabs, std::sqrt, std::sin, std::cos

#include "config.h"
#include "vec3.h"
#include "vec4.h"

BEGIN_NAMESPACE_PT

/**
 * @brief 4x4 矩阵类，用于3D变换，支持CUDA (__host__ __device__).
 *
 * - 支持标准矩阵运算：加减乘除、矩阵乘法等
 * - 提供矩阵变换操作：转置、求逆、单位矩阵等
 * - 支持将矩阵应用于 vec3/vec4：可变换点或方向向量
 * - 提供静态方法生成常用变换矩阵：平移、缩放、绕任意轴旋转（角度为弧度）
 * - 内部采用行优先存储 (row-major)，使用列向量乘法约定 (v' = M * v)
 */
struct mat4 {
public:
    float m[16]; ///< 内部以长度16数组存储矩阵元素（row-major 顺序）

    /** @name 构造函数 */
    ///@{
    PT_CPU_GPU mat4();                                       ///< 默认构造函数：初始化为单位矩阵
    PT_CPU_GPU mat4(float diagonal);                         ///< 将对角线初始化为给定值的矩阵（其余元素为0）
    PT_CPU_GPU mat4(const mat4& other)            = default; ///< 拷贝构造（支持 GPU）
    PT_CPU_GPU mat4& operator=(const mat4& other) = default; ///< 赋值操作符（支持 GPU）

    /// 用16个浮点数初始化矩阵（按行主序提供 m00...m33）
    PT_CPU_GPU mat4(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22,
                    float m23, float m30, float m31, float m32, float m33);
    /// 从包含16个浮点数的数组初始化矩阵（按行主序）
    PT_CPU_GPU explicit mat4(const float values[16]);
    ///@}

    /** @name 常用特殊矩阵的静态构造 */
    ///@{
    PT_CPU_GPU static mat4 identity();                                                          ///< 返回单位矩阵 I
    PT_CPU_GPU static mat4 zero();                                                              ///< 返回全零矩阵
    PT_CPU_GPU static mat4 translation(float tx, float ty, float tz);                           ///< 平移矩阵（沿x,y,z方向平移）
    PT_CPU_GPU static mat4 translation(const vec3& t);                                          ///< 平移矩阵（使用 vec3 提供的位移量）
    PT_CPU_GPU static mat4 scaling(float sx, float sy, float sz);                               ///< 缩放矩阵（各方向缩放系数）
    PT_CPU_GPU static mat4 scaling(const vec3& s);                                              ///< 缩放矩阵（使用 vec3 提供各方向缩放系数）
    PT_CPU_GPU static mat4 rotation(float angleRadians, const vec3& axis);                      ///< 绕任意轴旋转的矩阵（angle 为弧度）
    PT_CPU_GPU static mat4 rotation(float angleRadians, float axisX, float axisY, float axisZ); ///< 绕指定轴 (axisX, axisY, axisZ) 旋转的矩阵
    ///@}

    /** @name 元素访问 */
    ///@{
    PT_CPU_GPU float& operator()(int row, int col);             ///< 访问矩阵第(row行, col列)元素的引用
    PT_CPU_GPU const float& operator()(int row, int col) const; ///< 常量访问矩阵元素
    PT_CPU_GPU float* data();                                   ///< 返回指向内部数据的指针（float[16]，行主序）
    PT_CPU_GPU const float* data() const;                       ///< 常量版本的 data()
    ///@}

    /** @name 比较操作 */
    ///@{
    PT_CPU_GPU bool operator==(const mat4& rhs) const; ///< 精确比较两个矩阵是否相等
    PT_CPU_GPU bool operator!=(const mat4& rhs) const; ///< 不等比较
    ///@}

    /** @name 矩阵与矩阵的算术运算 */
    ///@{
    PT_CPU_GPU mat4 operator+(const mat4& rhs) const; ///< 矩阵加法
    PT_CPU_GPU mat4 operator-(const mat4& rhs) const; ///< 矩阵减法
    PT_CPU_GPU mat4 operator*(const mat4& rhs) const; ///< 矩阵乘法 (4x4 * 4x4)
    PT_CPU_GPU mat4& operator+=(const mat4& rhs);     ///< 加后赋值
    PT_CPU_GPU mat4& operator-=(const mat4& rhs);     ///< 减后赋值
    PT_CPU_GPU mat4& operator*=(const mat4& rhs);     ///< 乘后赋值 (this = this * rhs)
    ///@}

    /** @name 矩阵与标量的算术运算 */
    ///@{
    PT_CPU_GPU mat4 operator*(float scalar) const; ///< 矩阵乘以标量
    PT_CPU_GPU mat4 operator/(float scalar) const; ///< 矩阵除以标量
    PT_CPU_GPU mat4& operator*=(float scalar);     ///< 矩阵乘以标量并赋值
    PT_CPU_GPU mat4& operator/=(float scalar);     ///< 矩阵除以标量并赋值
    ///@}

    /** @name 一元运算符 */
    ///@{
    PT_CPU_GPU mat4 operator-() const; ///< 矩阵取负
    ///@}

    /** @name 矩阵性质与变换 */
    ///@{
    PT_CPU_GPU mat4 transposed() const; ///< 返回此矩阵的转置矩阵
    PT_CPU_GPU mat4 inverse() const;    ///< 返回此矩阵的逆矩阵（若不可逆将返回单位矩阵）
    ///@}

    /** @name 向量变换操作 */
    ///@{
    PT_CPU_GPU vec3 transformPoint(const vec3& v) const;  ///< 变换3D点向量 (vec3, w=1) ，会受到平移影响
    PT_CPU_GPU vec3 transformVector(const vec3& v) const; ///< 变换3D方向向量 (vec3, w=0) ，忽略平移分量
    PT_CPU_GPU vec4 operator*(const vec4& v) const;       ///< 矩阵与4D向量相乘
    PT_CPU_GPU vec3 operator*(const vec3& v) const;       ///< 矩阵与3D向量相乘（将vec3视为点，w=1）
    ///@}

    PT_CPU_GPU mat4& translate(const vec3& t);
    PT_CPU_GPU mat4& rotate(float angleDegrees, const vec3& axis);

    // 友元：实现标量 * 矩阵的交换律
    PT_CPU_GPU friend mat4 operator*(float scalar, const mat4& m) { return m * scalar; }
};

// ======================= 实现 =======================

PT_CPU_GPU inline mat4::mat4() {
    // 初始化为单位矩阵（对角线为1，其余为0）
    for (int i = 0; i < 16; ++i) { m[i] = (i % 5 == 0) ? 1.0f : 0.0f; }
}

PT_CPU_GPU inline mat4::mat4(float diagonal) {
    for (int i = 0; i < 16; ++i) { m[i] = 0.0f; }
    m[0] = m[5] = m[10] = m[15] = diagonal;
}

PT_CPU_GPU inline mat4::mat4(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22,
                             float m23, float m30, float m31, float m32, float m33) {
    m[0]  = m00;
    m[1]  = m01;
    m[2]  = m02;
    m[3]  = m03;
    m[4]  = m10;
    m[5]  = m11;
    m[6]  = m12;
    m[7]  = m13;
    m[8]  = m20;
    m[9]  = m21;
    m[10] = m22;
    m[11] = m23;
    m[12] = m30;
    m[13] = m31;
    m[14] = m32;
    m[15] = m33;
}

PT_CPU_GPU inline mat4::mat4(const float values[16]) {
    for (int i = 0; i < 16; ++i) { m[i] = values[i]; }
}

// 静态工厂函数实现
PT_CPU_GPU inline mat4 mat4::identity() {
    return mat4(); // 默认构造即为单位矩阵
}

PT_CPU_GPU inline mat4 mat4::zero() {
    mat4 result;
    for (int i = 0; i < 16; ++i) { result.m[i] = 0.0f; }
    return result;
}

PT_CPU_GPU inline mat4 mat4::translation(float tx, float ty, float tz) {
    mat4 result = mat4(); // 从单位矩阵开始
    // 设置平移分量（放入最后一列的前三个元素）
    result.m[3]  = tx;
    result.m[7]  = ty;
    result.m[11] = tz;
    // m[15] 保持为1（单位矩阵），其余非对角线元素保持0
    return result;
}

PT_CPU_GPU inline mat4 mat4::translation(const vec3& t) {
    // 假定 vec3 提供 x, y, z 成员
    return translation(t.x, t.y, t.z);
}

PT_CPU_GPU inline mat4 mat4::scaling(float sx, float sy, float sz) {
    mat4 result  = mat4();
    result.m[0]  = sx;
    result.m[5]  = sy;
    result.m[10] = sz;
    // m[15] 默认为1，其他已被构造函数置0
    return result;
}

PT_CPU_GPU inline mat4 mat4::scaling(const vec3& s) { return scaling(s.x, s.y, s.z); }

PT_CPU_GPU inline mat4 mat4::rotation(float angleRadians, float ax, float ay, float az) {
    // 绕任意轴 (ax, ay, az) 旋转 angleRadians 弧度的矩阵
    // 归一化轴向量：
    float len = std::sqrt(ax * ax + ay * ay + az * az);
    if (len < 1e-8f) {
        return mat4(); // 轴长度过小，返回单位矩阵
    }
    float x           = ax / len;
    float y           = ay / len;
    float z           = az / len;
    float c           = cosf(angleRadians);
    float s           = sinf(angleRadians);
    float one_minus_c = 1.0f - c;
    mat4 result;
    // 3x3 旋转部分元素（参考 Rodrigues 公式）
    result.m[0]  = c + one_minus_c * x * x;
    result.m[1]  = one_minus_c * x * y - s * z;
    result.m[2]  = one_minus_c * x * z + s * y;
    result.m[3]  = 0.0f;
    result.m[4]  = one_minus_c * y * x + s * z;
    result.m[5]  = c + one_minus_c * y * y;
    result.m[6]  = one_minus_c * y * z - s * x;
    result.m[7]  = 0.0f;
    result.m[8]  = one_minus_c * z * x - s * y;
    result.m[9]  = one_minus_c * z * y + s * x;
    result.m[10] = c + one_minus_c * z * z;
    result.m[11] = 0.0f;
    // 最后一行和最后一列使矩阵成为齐次坐标变换矩阵
    result.m[12] = result.m[13] = result.m[14] = 0.0f;
    result.m[15]                               = 1.0f;
    return result;
}

PT_CPU_GPU inline mat4 mat4::rotation(float angleRadians, const vec3& axis) { return rotation(angleRadians, axis.x, axis.y, axis.z); }

// 元素访问
PT_CPU_GPU inline float& mat4::operator()(int row, int col) { return m[row * 4 + col]; }

PT_CPU_GPU inline const float& mat4::operator()(int row, int col) const { return m[row * 4 + col]; }

PT_CPU_GPU inline float* mat4::data() { return m; }

PT_CPU_GPU inline const float* mat4::data() const { return m; }

// 比较操作
PT_CPU_GPU inline bool mat4::operator==(const mat4& rhs) const {
    for (int i = 0; i < 16; ++i) {
        if (m[i] != rhs.m[i]) { return false; }
    }
    return true;
}

PT_CPU_GPU inline bool mat4::operator!=(const mat4& rhs) const { return !(*this == rhs); }

// 矩阵加法
PT_CPU_GPU inline mat4 mat4::operator+(const mat4& rhs) const {
    mat4 result;
    for (int i = 0; i < 16; ++i) { result.m[i] = m[i] + rhs.m[i]; }
    return result;
}

// 矩阵减法
PT_CPU_GPU inline mat4 mat4::operator-(const mat4& rhs) const {
    mat4 result;
    for (int i = 0; i < 16; ++i) { result.m[i] = m[i] - rhs.m[i]; }
    return result;
}

// 矩阵乘法 (4x4 * 4x4)
PT_CPU_GPU inline mat4 mat4::operator*(const mat4& rhs) const {
    mat4 result;
    // 计算 result(i,j) = Σ_k this(i,k) * rhs(k,j)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) { sum += (*this)(i, k) * rhs(k, j); }
            result(i, j) = sum;
        }
    }
    return result;
}

PT_CPU_GPU inline mat4& mat4::operator+=(const mat4& rhs) {
    for (int i = 0; i < 16; ++i) { m[i] += rhs.m[i]; }
    return *this;
}

PT_CPU_GPU inline mat4& mat4::operator-=(const mat4& rhs) {
    for (int i = 0; i < 16; ++i) { m[i] -= rhs.m[i]; }
    return *this;
}

PT_CPU_GPU inline mat4& mat4::operator*=(const mat4& rhs) {
    *this = *this * rhs;
    return *this;
}

// 标量乘法
PT_CPU_GPU inline mat4 mat4::operator*(float scalar) const {
    mat4 result;
    for (int i = 0; i < 16; ++i) { result.m[i] = m[i] * scalar; }
    return result;
}

// 标量除法
PT_CPU_GPU inline mat4 mat4::operator/(float scalar) const {
    mat4 result;
    float inv = 1.0f / scalar;
    for (int i = 0; i < 16; ++i) { result.m[i] = m[i] * inv; }
    return result;
}

PT_CPU_GPU inline mat4& mat4::operator*=(float scalar) {
    for (int i = 0; i < 16; ++i) { m[i] *= scalar; }
    return *this;
}

PT_CPU_GPU inline mat4& mat4::operator/=(float scalar) {
    float inv = 1.0f / scalar;
    for (int i = 0; i < 16; ++i) { m[i] *= inv; }
    return *this;
}

// 一元取负
PT_CPU_GPU inline mat4 mat4::operator-() const {
    mat4 result;
    for (int i = 0; i < 16; ++i) { result.m[i] = -m[i]; }
    return result;
}

// 转置矩阵
PT_CPU_GPU inline mat4 mat4::transposed() const {
    mat4 result;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) { result(j, i) = (*this)(i, j); }
    }
    return result;
}

// 求逆矩阵（高斯-约当消元法）
PT_CPU_GPU inline mat4 mat4::inverse() const {
    mat4 inv = mat4::identity();
    mat4 A   = *this;
    // 依次对每一列进行消元
    for (int i = 0; i < 4; ++i) {
        // 寻找第 i 列绝对值最大的主元
        float pivotValue = std::fabs(A(i, i));
        int pivotRow     = i;
        for (int j = i + 1; j < 4; ++j) {
            float candidate = std::fabs(A(j, i));
            if (candidate > pivotValue) {
                pivotValue = candidate;
                pivotRow   = j;
            }
        }
        if (pivotValue < 1e-8f) {
            // 近似不可逆，返回单位矩阵以避免计算错误
            return mat4::identity();
        }
        // 如果找到更大的主元则交换行
        if (pivotRow != i) {
            for (int k = 0; k < 4; ++k) {
                std::swap(A(i, k), A(pivotRow, k));
                std::swap(inv(i, k), inv(pivotRow, k));
            }
        }
        // 将主元所在行标准化（主元归一化为1）
        float pivot = A(i, i);
        for (int k = 0; k < 4; ++k) {
            A(i, k) /= pivot;
            inv(i, k) /= pivot;
        }
        // 对其他行消去该列元素
        for (int j = 0; j < 4; ++j) {
            if (j != i) {
                float factor = A(j, i);
                for (int k = 0; k < 4; ++k) {
                    A(j, k) -= factor * A(i, k);
                    inv(j, k) -= factor * inv(i, k);
                }
            }
        }
    }
    return inv;
}

// 向量变换
PT_CPU_GPU inline vec3 mat4::transformPoint(const vec3& v) const {
    // 将 vec3 按 (x, y, z, 1) 处理
    float x = v.x;
    float y = v.y;
    float z = v.z;
    vec3 result;
    // 计算变换后坐标：包括平移分量
    result.x = m[0] * x + m[1] * y + m[2] * z + m[3];
    result.y = m[4] * x + m[5] * y + m[6] * z + m[7];
    result.z = m[8] * x + m[9] * y + m[10] * z + m[11];
    return result;
}

PT_CPU_GPU inline vec3 mat4::transformVector(const vec3& v) const {
    // 将 vec3 按 (x, y, z, 0) 处理（忽略平移）
    float x = v.x;
    float y = v.y;
    float z = v.z;
    vec3 result;
    result.x = m[0] * x + m[1] * y + m[2] * z;
    result.y = m[4] * x + m[5] * y + m[6] * z;
    result.z = m[8] * x + m[9] * y + m[10] * z;
    return result;
}

PT_CPU_GPU inline vec4 mat4::operator*(const vec4& v) const {
    // 4x4 矩阵 * 4D向量
    float x = v.x;
    float y = v.y;
    float z = v.z;
    float w = v.w;
    vec4 result;
    result.x = m[0] * x + m[1] * y + m[2] * z + m[3] * w;
    result.y = m[4] * x + m[5] * y + m[6] * z + m[7] * w;
    result.z = m[8] * x + m[9] * y + m[10] * z + m[11] * w;
    result.w = m[12] * x + m[13] * y + m[14] * z + m[15] * w;
    return result;
}

PT_CPU_GPU inline vec3 mat4::operator*(const vec3& v) const {
    // 4x4 矩阵 * 3D向量（默认为点变换，w=1）
    return transformPoint(v);
}

PT_CPU_GPU inline mat4& mat4::translate(const vec3& t) {
    // 假定右乘变换（v' = M * v），平移矩阵放在右侧
    *this = (*this) * mat4::translation(t);
    return *this;
}

PT_CPU_GPU inline mat4& mat4::rotate(float angleDegrees, const vec3& axis) {
    // 将角度转换为弧度
    const float PI     = 3.14159265358979323846f;
    float angleRadians = angleDegrees * PI / 180.0f;
    *this              = (*this) * mat4::rotation(angleRadians, axis);
    return *this;
}

END_NAMESPACE_PT
