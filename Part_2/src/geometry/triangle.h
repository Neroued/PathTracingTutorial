#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/bounds.h"
#include "math/math_utils.h"
#include "core/interaction.h"
#include "core/ray.h"
#include "geometry/vertex.h"

namespace pt {

PT_HD inline SurfaceInteraction intersectTriangle(
    const Ray& ray,
    const Vertex& va, const Vertex& vb, const Vertex& vc,
    int materialID)
{
    SurfaceInteraction si;

    vec3 edge1 = vb.position - va.position;
    vec3 edge2 = vc.position - va.position;
    vec3 pvec  = cross(ray.direction, edge2);
    float det  = dot(edge1, pvec);

    if (pt::abs(det) < Epsilon) return si;

    float invDet = 1.0f / det;
    vec3 tvec    = ray.origin - va.position;

    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return si;

    vec3 qvec = cross(tvec, edge1);
    float v   = dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return si;

    float t = dot(edge2, qvec) * invDet;
    if (t < 1e-4f) return si;

    float w = 1.0f - u - v;

    si.hit        = true;
    si.distance   = t;
    si.point      = ray.origin + t * ray.direction;
    si.normal     = normalize(w * va.normal + u * vb.normal + v * vc.normal);
    si.uv         = w * va.uv + u * vb.uv + v * vc.uv;
    si.materialId = materialID;
    si.baryU      = u;
    si.baryV      = v;
    return si;
}

PT_HD inline float intersectTriangleShadow(
    const Ray& ray,
    const Vertex& va, const Vertex& vb, const Vertex& vc)
{
    vec3 edge1 = vb.position - va.position;
    vec3 edge2 = vc.position - va.position;
    vec3 pvec  = cross(ray.direction, edge2);
    float det  = dot(edge1, pvec);

    if (pt::abs(det) < Epsilon) return -1.0f;

    float invDet = 1.0f / det;
    vec3 tvec    = ray.origin - va.position;

    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return -1.0f;

    vec3 qvec = cross(tvec, edge1);
    float v   = dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;

    float t = dot(edge2, qvec) * invDet;
    return t > 1e-4f ? t : -1.0f;
}

PT_HD inline Bounds computeFaceBounds(const Vertex* vertices, const TriangleFace& face) {
    Bounds b;
    b.expand(vertices[face.v0].position);
    b.expand(vertices[face.v1].position);
    b.expand(vertices[face.v2].position);
    return b;
}

} // namespace pt
