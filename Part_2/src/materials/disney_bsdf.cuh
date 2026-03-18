#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/math_utils.h"
#include "math/frame.h"
#include "math/sampling.h"
#include "materials/material.h"
#include "materials/microfacet.cuh"

#include <cuda_runtime.h>
#include <texture_indirect_functions.h>

namespace pt {

PT_HD inline float luminance_bsdf(const vec3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// Resolved material parameters after texture lookups
struct MaterialEval {
    vec3  baseColor;
    float metallic, roughness, specTrans, ior;
    float clearcoat, clearcoatRoughness;
    float sheen, sheenTint, subsurface;
    float anisotropic, specularTint;
    float specularFactor;
    vec3  specularColor;
    vec3  emissive;
    float emissiveStrength;
    vec3  sheenColor;
    float anisotropyRotation;
    float thicknessFactor;
    float attenuationDistance;
    vec3  attenuationColor;
};

struct BsdfEval {
    vec3  f   = vec3(0.0f);
    float pdf = 0.0f;
};

struct BsdfSample {
    vec3  wi     = vec3(0.0f);
    vec3  weight = vec3(0.0f);
    float pdf    = 0.0f;
    bool  delta  = false;
    bool  valid  = false;
};

// ---- Texture fetch ----

__device__ inline MaterialEval fetchMaterial(const Material& mat, const vec2& uv,
                                             const cudaTextureObject_t* textures,
                                             uint32_t numTextures) {
    MaterialEval m;
    m.baseColor          = mat.baseColor;
    m.metallic           = mat.metallic;
    m.roughness          = mat.roughness;
    m.specTrans          = mat.specTrans;
    m.ior                = mat.ior;
    m.clearcoat          = mat.clearcoat;
    m.clearcoatRoughness = mat.clearcoatRoughness;
    m.sheen              = mat.sheen;
    m.sheenTint          = mat.sheenTint;
    m.subsurface         = mat.subsurface;
    m.anisotropic        = mat.anisotropic;
    m.specularTint       = mat.specularTint;
    m.specularFactor     = mat.specularFactor;
    m.specularColor      = vec3(mat.specularColorR, mat.specularColorG, mat.specularColorB);
    m.emissive           = mat.emissive;
    m.emissiveStrength   = mat.emissiveStrength;
    m.sheenColor         = vec3(mat.sheenColorR, mat.sheenColorG, mat.sheenColorB);
    m.anisotropyRotation = mat.anisotropyRotation;
    m.thicknessFactor    = mat.thicknessFactor;
    m.attenuationDistance = mat.attenuationDistance;
    m.attenuationColor   = vec3(mat.attColorR, mat.attColorG, mat.attColorB);

    if (textures && numTextures > 0) {
        if (mat.baseColorTexId >= 0 && static_cast<uint32_t>(mat.baseColorTexId) < numTextures) {
            float4 tc = tex2D<float4>(textures[mat.baseColorTexId], uv.x, uv.y);
            vec3 texColor(pt::pow(tc.x, 2.2f), pt::pow(tc.y, 2.2f), pt::pow(tc.z, 2.2f));
            m.baseColor = m.baseColor * texColor;
        }
        if (mat.metallicRoughnessTexId >= 0 && static_cast<uint32_t>(mat.metallicRoughnessTexId) < numTextures) {
            float4 mr = tex2D<float4>(textures[mat.metallicRoughnessTexId], uv.x, uv.y);
            m.roughness *= mr.y;   // green channel
            m.metallic  *= mr.z;   // blue channel
        }
        if (mat.emissiveTexId >= 0 && static_cast<uint32_t>(mat.emissiveTexId) < numTextures) {
            float4 em = tex2D<float4>(textures[mat.emissiveTexId], uv.x, uv.y);
            vec3 emColor(pt::pow(em.x, 2.2f), pt::pow(em.y, 2.2f), pt::pow(em.z, 2.2f));
            m.emissive = m.emissive * emColor;
        }
        if (mat.clearcoatTexId >= 0 && static_cast<uint32_t>(mat.clearcoatTexId) < numTextures) {
            float4 cc = tex2D<float4>(textures[mat.clearcoatTexId], uv.x, uv.y);
            m.clearcoat *= cc.x;   // red channel
        }
        if (mat.clearcoatRoughnessTexId >= 0 && static_cast<uint32_t>(mat.clearcoatRoughnessTexId) < numTextures) {
            float4 cr = tex2D<float4>(textures[mat.clearcoatRoughnessTexId], uv.x, uv.y);
            m.clearcoatRoughness *= cr.y;   // green channel
        }
        if (mat.transmissionTexId >= 0 && static_cast<uint32_t>(mat.transmissionTexId) < numTextures) {
            float4 tr = tex2D<float4>(textures[mat.transmissionTexId], uv.x, uv.y);
            m.specTrans *= tr.x;   // red channel
        }
        if (mat.sheenColorTexId >= 0 && static_cast<uint32_t>(mat.sheenColorTexId) < numTextures) {
            float4 sc = tex2D<float4>(textures[mat.sheenColorTexId], uv.x, uv.y);
            vec3 scColor(pt::pow(sc.x, 2.2f), pt::pow(sc.y, 2.2f), pt::pow(sc.z, 2.2f));
            m.sheenColor = m.sheenColor * scColor;
        }
        if (mat.sheenRoughnessTexId >= 0 && static_cast<uint32_t>(mat.sheenRoughnessTexId) < numTextures) {
            float4 sr = tex2D<float4>(textures[mat.sheenRoughnessTexId], uv.x, uv.y);
            m.sheen *= sr.w;   // alpha channel per spec
        }
    }

    m.roughness          = pt::clamp(m.roughness, 0.001f, 1.0f);
    m.metallic           = pt::clamp(m.metallic, 0.0f, 1.0f);
    m.clearcoatRoughness = pt::clamp(m.clearcoatRoughness, 0.001f, 1.0f);
    return m;
}

// ---- Helper: anisotropic alpha from roughness + anisotropic param ----

__device__ inline void computeAlpha(float roughness, float anisotropic, float& ax, float& ay) {
    float aspect = pt::sqrt(1.0f - 0.9f * anisotropic);
    float a = roughness * roughness;
    ax = pt::max(a / aspect, 0.001f);
    ay = pt::max(a * aspect, 0.001f);
}

// ---- Disney Diffuse (Burley 2012) ----

__device__ inline vec3 evalDisneyDiffuse(const MaterialEval& m, float NoL, float NoV,
                                          float HoL) {
    float fd90 = 0.5f + 2.0f * HoL * HoL * m.roughness;
    float fL   = F_Schlick(NoL, 1.0f) * (fd90 - 1.0f) + 1.0f;
    float fV   = F_Schlick(NoV, 1.0f) * (fd90 - 1.0f) + 1.0f;

    // Approximate retro-reflection via Schlick on the "light" and "view" directions
    // Note: F_Schlick(x, 1.0f) = 1.0f always, so we compute the Disney fd scaling manually
    float flSchlick = 1.0f + (fd90 - 1.0f) * pt::pow(1.0f - NoL, 5.0f);
    float fvSchlick = 1.0f + (fd90 - 1.0f) * pt::pow(1.0f - NoV, 5.0f);

    vec3 diffuse = m.baseColor * (InvPi * flSchlick * fvSchlick);

    // Subsurface approximation (Hanrahan-Krueger inspired)
    if (m.subsurface > 0.0f) {
        float fss90 = HoL * HoL * m.roughness;
        float fssL  = 1.0f + (fss90 - 1.0f) * pt::pow(1.0f - NoL, 5.0f);
        float fssV  = 1.0f + (fss90 - 1.0f) * pt::pow(1.0f - NoV, 5.0f);
        float ss    = 1.25f * (fssL * fssV * (1.0f / (NoL + NoV + 1e-5f) - 0.5f) + 0.5f);
        vec3 subsurfDiff = m.baseColor * (InvPi * ss);
        diffuse = (1.0f - m.subsurface) * diffuse + m.subsurface * subsurfDiff;
    }

    return diffuse;
}

// ---- Sheen ----

__device__ inline vec3 evalSheen(const MaterialEval& m, float HoL) {
    if (m.sheen <= 0.0f) return vec3(0.0f);
    float lum = luminance_bsdf(m.baseColor);
    vec3 tint = lum > 0.0f ? m.baseColor / lum : vec3(1.0f);
    vec3 baseSheenColor = (1.0f - m.sheenTint) * vec3(1.0f) + m.sheenTint * tint;
    vec3 finalSheenColor = baseSheenColor * m.sheenColor;
    float fh = pt::pow(1.0f - HoL, 5.0f);
    return m.sheen * finalSheenColor * fh;
}

// ---- Specular F0 ----

__device__ inline vec3 computeF0(const MaterialEval& m) {
    float lum = luminance_bsdf(m.baseColor);
    vec3 tint = lum > 0.0f ? m.baseColor / lum : vec3(1.0f);
    float eta = m.ior;
    float r0  = ((eta - 1.0f) * (eta - 1.0f)) / ((eta + 1.0f) * (eta + 1.0f));
    vec3 spec = r0 * ((1.0f - m.specularTint) * vec3(1.0f) + m.specularTint * tint);
    spec = spec * m.specularFactor * m.specularColor;
    return (1.0f - m.metallic) * spec + m.metallic * m.baseColor;
}

// ---- Lobe weight computation ----

__device__ inline void computeLobeWeights(const MaterialEval& m,
                                           float& wDiffuse, float& wSpec,
                                           float& wClearcoat, float& wTrans,
                                           float& wSheen) {
    float diffWeight = (1.0f - m.metallic) * (1.0f - m.specTrans);
    float specWeight = 1.0f;
    float transWeight = (1.0f - m.metallic) * m.specTrans;
    float ccWeight   = 0.25f * m.clearcoat;
    float sheenWeight = (1.0f - m.metallic) * m.sheen * 0.1f;

    float total = diffWeight + specWeight + ccWeight + transWeight + sheenWeight;
    if (total <= 0.0f) { wDiffuse = wSpec = wClearcoat = wTrans = wSheen = 0.0f; return; }
    float inv = 1.0f / total;
    wDiffuse   = diffWeight * inv;
    wSpec      = specWeight * inv;
    wClearcoat = ccWeight * inv;
    wTrans     = transWeight * inv;
    wSheen     = sheenWeight * inv;
}

// ---- Eval: evaluate full Disney BSDF for a given (wo, wi) pair ----

__device__ inline BsdfEval evalDisneyBsdf(const MaterialEval& m,
                                           const vec3& wo, const vec3& wi,
                                           const vec3& N, const vec3& T,
                                           bool entering = true) {
    BsdfEval result;
    float NoL = dot(N, wi);
    float NoV = dot(N, wo);

    bool reflect = NoL > 0.0f;

    // Transmission: wi is on the opposite side
    if (!reflect && m.specTrans <= 0.0f) return result;
    if (!reflect) {
        // Evaluate transmission lobe only
        float absNoL = pt::abs(NoL);
        float absNoV = pt::abs(NoV);
        if (absNoL < 1e-5f || absNoV < 1e-5f) return result;

        float eta = entering ? (1.0f / m.ior) : m.ior;
        vec3 ht = -normalize(wo * eta + wi);
        float HoV = pt::abs(dot(ht, wo));

        float ax, ay;
        computeAlpha(m.roughness, m.anisotropic, ax, ay);
        Frame frame(N);
        vec3 ht_local = frame.toLocal(ht);
        float D = D_GGX_Aniso(pt::abs(ht_local.z), ht_local.x, ht_local.y, ax, ay);
        float G = G2_SmithCorrelated(absNoL, absNoV, pt::sqrt(ax * ay));
        float F = F_Dielectric(HoV, eta);

        float sqrtDenom = dot(wo, ht) * eta + dot(wi, ht);
        float factor = pt::abs(dot(wo, ht)) * pt::abs(dot(wi, ht))
                     / (absNoV * absNoL * sqrtDenom * sqrtDenom + 1e-7f);

        float transWeight = (1.0f - m.metallic) * m.specTrans;
        vec3 ftrans = m.baseColor * transWeight * D * G * (1.0f - F) * factor;

        float wD, wS, wC, wT, wSh;
        computeLobeWeights(m, wD, wS, wC, wT, wSh);
        float jacFactor = eta * eta * pt::abs(dot(wi, ht))
                        / (sqrtDenom * sqrtDenom + 1e-7f);
        float transPdf = D * G1_Smith(absNoV, pt::sqrt(ax * ay)) * HoV / (absNoV + 1e-7f) * jacFactor;

        result.f   = ftrans;
        result.pdf = wT * transPdf;
        return result;
    }

    // Reflection hemisphere
    if (NoL < 1e-5f || NoV < 1e-5f) return result;

    vec3 H = normalize(wo + wi);
    float NoH = pt::max(dot(N, H), 0.0f);
    float HoL = pt::max(dot(H, wi), 0.0f);

    Frame frame(N);
    vec3 h_local = frame.toLocal(H);

    float ax, ay;
    computeAlpha(m.roughness, m.anisotropic, ax, ay);

    float wDiff, wSpec, wCC, wTrans, wSheen;
    computeLobeWeights(m, wDiff, wSpec, wCC, wTrans, wSheen);

    vec3 f_total(0.0f);
    float pdf_total = 0.0f;

    // Diffuse
    float diffWeight = (1.0f - m.metallic) * (1.0f - m.specTrans);
    if (diffWeight > 0.0f) {
        vec3 fd = evalDisneyDiffuse(m, NoL, NoV, HoL);
        f_total += diffWeight * fd;
        pdf_total += wDiff * (NoL * InvPi);
    }

    // Sheen
    if (m.sheen > 0.0f && m.metallic < 1.0f) {
        vec3 fs = evalSheen(m, HoL);
        f_total += (1.0f - m.metallic) * fs;
        pdf_total += wSheen * (NoL * InvPi);
    }

    // Specular (GGX)
    {
        float D = D_GGX_Aniso(NoH, h_local.x, h_local.y, ax, ay);
        float G = G2_SmithCorrelated(NoL, NoV, pt::sqrt(ax * ay));
        vec3 F0 = computeF0(m);
        vec3 F  = F_Schlick(HoL, F0);
        vec3 spec = D * G * F / (4.0f * NoL * NoV + 1e-7f);
        f_total += spec;

        vec3 wo_local = frame.toLocal(wo);
        float specPdf = pdfVNDF(wo_local, h_local, ax, ay) / (4.0f * HoL + 1e-7f);
        pdf_total += wSpec * specPdf;
    }

    // Clearcoat
    if (m.clearcoat > 0.0f) {
        float ccAlpha = pt::max(m.clearcoatRoughness * m.clearcoatRoughness, 0.001f);
        float D_cc = D_GGX(NoH, ccAlpha);
        float G_cc = G2_SmithCorrelated(NoL, NoV, ccAlpha);
        float F_cc = F_Schlick(HoL, 0.04f);
        float ccSpec = 0.25f * m.clearcoat * D_cc * G_cc * F_cc / (4.0f * NoL * NoV + 1e-7f);
        f_total += vec3(ccSpec);

        float ccPdf = D_cc * NoH / (4.0f * HoL + 1e-7f);
        pdf_total += wCC * ccPdf;
    }

    result.f   = f_total * NoL;
    result.pdf = pdf_total;
    return result;
}

// ---- Sample: importance-sample one lobe of the Disney BSDF ----

template <typename Rng>
__device__ inline BsdfSample sampleDisneyBsdf(const MaterialEval& m,
                                                const vec3& wo, const vec3& N,
                                                const vec3& T,
                                                Rng& rng,
                                                bool entering = true) {
    BsdfSample result;
    float NoV = dot(N, wo);
    if (NoV < 1e-5f) return result;

    float wDiff, wSpec, wCC, wTrans, wSheen;
    computeLobeWeights(m, wDiff, wSpec, wCC, wTrans, wSheen);

    float choice = rng.get1D();

    Frame frame(N);
    vec3 wo_local = frame.toLocal(wo);

    float ax, ay;
    computeAlpha(m.roughness, m.anisotropic, ax, ay);

    vec3 wi;
    bool isDelta = false;

    if (choice < wDiff + wSheen) {
        vec3 localDir = sampleCosineHemisphere(rng.get1D(), rng.get1D());
        wi = frame.toWorld(localDir);
    }
    else if (choice < wDiff + wSheen + wSpec) {
        vec3 h_local = sampleVNDF(wo_local, ax, ay, rng.get1D(), rng.get1D());
        vec3 H = frame.toWorld(h_local);
        wi = reflect(-wo, H);
        if (dot(wi, N) <= 0.0f) return result;
        isDelta = (m.roughness < 0.01f);
    }
    else if (choice < wDiff + wSheen + wSpec + wCC) {
        float ccAlpha = pt::max(m.clearcoatRoughness * m.clearcoatRoughness, 0.001f);
        vec3 h_local = sampleVNDF(wo_local, ccAlpha, ccAlpha, rng.get1D(), rng.get1D());
        vec3 H = frame.toWorld(h_local);
        wi = reflect(-wo, H);
        if (dot(wi, N) <= 0.0f) return result;
    }
    else {
        vec3 h_local = sampleVNDF(wo_local, ax, ay, rng.get1D(), rng.get1D());
        vec3 H = frame.toWorld(h_local);
        float HoV = dot(H, wo);
        float eta = entering ? (1.0f / m.ior) : m.ior;
        vec3 wt;
        if (!refractDir(wo, H, eta, wt)) {
            wi = reflect(-wo, H);
            if (dot(wi, N) <= 0.0f) return result;
        } else {
            wi = normalize(wt);
            isDelta = (m.roughness < 0.01f);
        }
    }

    // Evaluate the full BSDF at the sampled direction
    BsdfEval eval = evalDisneyBsdf(m, wo, wi, N, T, entering);
    if (eval.pdf <= 0.0f) return result;

    result.wi     = wi;
    result.pdf    = eval.pdf;
    result.delta  = isDelta;
    result.valid  = true;

    if (isDelta) {
        // For delta distributions, weight = f / pdf is not meaningful via eval
        // Compute analytic weight
        float NoL = pt::abs(dot(N, wi));
        bool isTransmission = dot(N, wi) < 0.0f;
        if (isTransmission) {
            result.weight = m.baseColor;
        } else {
            vec3 F0 = computeF0(m);
            vec3 H  = normalize(wo + wi);
            float HoL = pt::max(dot(H, wi), 0.0f);
            result.weight = F_Schlick(HoL, F0);
        }
    } else {
        float NoL = pt::abs(dot(N, wi));
        result.weight = eval.f / (eval.pdf + 1e-7f);
    }

    return result;
}

// ---- PDF: evaluate combined PDF for a given (wo, wi) pair ----

__device__ inline float pdfDisneyBsdf(const MaterialEval& m,
                                       const vec3& wo, const vec3& wi,
                                       const vec3& N, const vec3& T,
                                       bool entering = true) {
    BsdfEval eval = evalDisneyBsdf(m, wo, wi, N, T, entering);
    return eval.pdf;
}

} // namespace pt
