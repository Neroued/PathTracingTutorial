#pragma once

#include <QVector3D>

struct alignas(16) Primitive
{
    QVector3D v1;
    int primitiveType; // 0 for triangle, 1 for sphere
    QVector3D v2;
    int materialID;
    QVector3D v3;
    float fpad1;
    QVector3D v4;
    float fpad2;

    static Primitive CreateTriangle(const QVector3D &p1, const QVector3D &p2, const QVector3D &p3, const QVector3D &normal, int matID)
    {
        return Primitive{p1, 0, p2, matID, p3, 0.0f, normal, 0.0f};
    }

    static Primitive CreateSphere(const QVector3D &center, float radius, int matID)
    {
        return Primitive{center, 1, {}, matID, {}, radius, {}, 0.0f};
    }

private:
    // 私有构造函数，避免直接实例化
    Primitive(const QVector3D &_v1, int _primitiveType, const QVector3D &_v2, int _materialID,
              const QVector3D &_v3, float _fpad1, const QVector3D &_v4, float _fpad2)
        : v1(_v1), primitiveType(_primitiveType), v2(_v2), materialID(_materialID),
          v3(_v3), fpad1(_fpad1), v4(_v4), fpad2(_fpad2) {}
};


/* 使用一个统一的占用64字节的结构来存储Primitive
 * 使用primitiveType来判断存储的图元类型
 * 0 表示三角形
 * 1 表示球形
 */

// case: primitiveType == 0
// struct alignas(16) Triangle
// {
//     QVector3D p1;
//     int primitiveType = 0;
//     QVector3D p2;
//     int materialID;
//     QVector3D p3;
//     float fpad1;
//     QVector3D normal;
//     float fpad2;
// };

// case: primitiveType == 1
// struct alignas(16) Sphere
// {
//     QVector3D center;
//     int primitiveType = 1;
//     QVector3D vpad1;
//     int materialID;
//     QVector3D vpad2;
//     float radius;
//     QVector3D vpad3;
//     float fpad;
// };