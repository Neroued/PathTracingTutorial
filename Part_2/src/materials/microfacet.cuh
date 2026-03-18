#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/math_utils.h"

namespace pt {

// GGX (Trowbridge-Reitz) normal distribution function -- isotropic
PT_HD inline float D_GGX(float NoH, float alpha) {
    float a2 = alpha * alpha;
    float d  = NoH * NoH * (a2 - 1.0f) + 1.0f;
    return a2 / (Pi * d * d + 1e-7f);
}

// GGX NDF -- anisotropic
PT_HD inline float D_GGX_Aniso(float NoH, float HdotX, float HdotY, float ax, float ay) {
    float hx = HdotX / ax;
    float hy = HdotY / ay;
    float d  = hx * hx + hy * hy + NoH * NoH;
    return 1.0f / (Pi * ax * ay * d * d + 1e-7f);
}

// Smith G1 (GGX)
PT_HD inline float G1_Smith(float NoV, float alpha) {
    float a2 = alpha * alpha;
    return 2.0f * NoV / (NoV + pt::sqrt(a2 + (1.0f - a2) * NoV * NoV) + 1e-7f);
}

// Height-correlated Smith G2
PT_HD inline float G2_SmithCorrelated(float NoL, float NoV, float alpha) {
    float a2 = alpha * alpha;
    float gv = NoL * pt::sqrt(a2 + (1.0f - a2) * NoV * NoV);
    float gl = NoV * pt::sqrt(a2 + (1.0f - a2) * NoL * NoL);
    return 2.0f * NoL * NoV / (gv + gl + 1e-7f);
}

// Schlick Fresnel approximation (scalar)
PT_HD inline float F_Schlick(float cosTheta, float F0) {
    float t = pt::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float t2 = t * t;
    return F0 + (1.0f - F0) * t2 * t2 * t;
}

// Schlick Fresnel (vec3 for metals)
PT_HD inline vec3 F_Schlick(float cosTheta, const vec3& F0) {
    float t = pt::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float t2 = t * t;
    float t5 = t2 * t2 * t;
    return F0 + (vec3(1.0f) - F0) * t5;
}

// Exact dielectric Fresnel reflectance
PT_HD inline float F_Dielectric(float cosThetaI, float eta) {
    float sinThetaTSq = eta * eta * (1.0f - cosThetaI * cosThetaI);
    if (sinThetaTSq >= 1.0f) return 1.0f;
    float cosThetaT = pt::sqrt(pt::max(0.0f, 1.0f - sinThetaTSq));
    float ci = pt::abs(cosThetaI);
    float rs = (ci - eta * cosThetaT) / (ci + eta * cosThetaT + 1e-7f);
    float rp = (eta * ci - cosThetaT) / (eta * ci + cosThetaT + 1e-7f);
    return 0.5f * (rs * rs + rp * rp);
}

// VNDF sampling (Heitz 2018) -- supports anisotropic alpha
// wo_local: outgoing direction in local shading space (z = normal)
// Returns the sampled half-vector in local space
PT_HD inline vec3 sampleVNDF(const vec3& wo_local, float ax, float ay, float u1, float u2) {
    // Stretch wo
    vec3 wh = normalize(vec3(ax * wo_local.x, ay * wo_local.y, wo_local.z));
    if (wh.z < 0.0f) wh = -wh;

    // Build orthonormal basis around wh
    vec3 T1 = (wh.z < 0.9999f)
        ? normalize(cross(vec3(0,0,1), wh))
        : vec3(1,0,0);
    vec3 T2 = cross(wh, T1);

    // Parameterization of the projected area
    float r   = pt::sqrt(u1);
    float phi = TwoPi * u2;
    float t1  = r * pt::cos(phi);
    float t2  = r * pt::sin(phi);
    float s   = 0.5f * (1.0f + wh.z);
    t2 = (1.0f - s) * pt::sqrt(pt::max(0.0f, 1.0f - t1 * t1)) + s * t2;

    // Reproject onto hemisphere
    vec3 Nh = t1 * T1 + t2 * T2
            + pt::sqrt(pt::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * wh;

    // Un-stretch
    return normalize(vec3(ax * Nh.x, ay * Nh.y, pt::max(0.0f, Nh.z)));
}

// PDF of VNDF sampling
PT_HD inline float pdfVNDF(const vec3& wo_local, const vec3& h_local, float ax, float ay) {
    float NoV = pt::max(wo_local.z, 1e-5f);
    float NoH = pt::max(h_local.z, 0.0f);
    float VoH = pt::max(dot(wo_local, h_local), 0.0f);
    float D   = D_GGX_Aniso(NoH, h_local.x, h_local.y, ax, ay);
    float G1  = G1_Smith(NoV, pt::sqrt(ax * ay));
    return D * G1 * VoH / (NoV + 1e-7f);
}

// Refract direction (returns false for total internal reflection)
PT_HD inline bool refractDir(const vec3& wi, const vec3& n, float eta, vec3& wt) {
    float cosThetaI = dot(wi, n);
    float sin2ThetaI = pt::max(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;
    if (sin2ThetaT >= 1.0f) return false;
    float cosThetaT = pt::sqrt(1.0f - sin2ThetaT);
    wt = eta * (-wi) + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

} // namespace pt
