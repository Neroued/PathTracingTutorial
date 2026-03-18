#pragma once

#include "pt.h"
#include "math/vecmath.h"

namespace pt {

struct Material {
    // Disney Principled BSDF -- base layer
    vec3  baseColor          = {0.8f, 0.8f, 0.8f};
    float metallic           = 0.0f;

    float roughness          = 0.5f;
    float specularTint       = 0.0f;
    float anisotropic        = 0.0f;
    float sheen              = 0.0f;

    float sheenTint          = 0.5f;
    float clearcoat          = 0.0f;
    float clearcoatRoughness = 0.03f;
    float specTrans          = 0.0f;

    float ior                = 1.5f;
    float subsurface         = 0.0f;
    float specularFactor     = 1.0f;  // KHR_materials_specular
    float specularColorR     = 1.0f;

    float specularColorG     = 1.0f;
    float specularColorB     = 1.0f;
    float anisotropyRotation = 0.0f;  // KHR_materials_anisotropy
    float normalScale        = 1.0f;  // normalTexture.scale

    // Emission
    vec3  emissive           = {0.0f, 0.0f, 0.0f};
    float emissiveStrength   = 1.0f;

    // Alpha & surface flags
    float alphaCutoff        = 0.0f;
    int   alphaMode          = 0;   // 0 = OPAQUE, 1 = MASK, 2 = BLEND
    float baseAlpha          = 1.0f;
    int   doubleSided        = 0;

    // Sheen color (KHR_materials_sheen)
    float sheenColorR        = 1.0f;
    float sheenColorG        = 1.0f;
    float sheenColorB        = 1.0f;
    float occlusionStrength  = 1.0f;  // occlusionTexture.strength

    // Volume (KHR_materials_volume)
    float thicknessFactor    = 0.0f;
    float attenuationDistance = 1e30f;
    float attColorR          = 1.0f;
    float attColorG          = 1.0f;

    float attColorB          = 1.0f;
    // UV transform (KHR_texture_transform)
    float uvOffsetX          = 0.0f;
    float uvOffsetY          = 0.0f;
    float uvScaleX           = 1.0f;

    float uvScaleY           = 1.0f;
    float uvRotation         = 0.0f;
    float pad1_              = 0.0f;
    float pad2_              = 0.0f;

    // Texture indices (-1 = no texture)
    int   baseColorTexId          = -1;
    int   metallicRoughnessTexId  = -1;
    int   normalTexId             = -1;
    int   emissiveTexId           = -1;

    int   occlusionTexId          = -1;
    int   clearcoatTexId          = -1;
    int   clearcoatRoughnessTexId = -1;
    int   clearcoatNormalTexId    = -1;

    int   transmissionTexId       = -1;
    int   sheenColorTexId         = -1;
    int   sheenRoughnessTexId     = -1;
    int   pad3_                   = 0;
};

PT_HD inline bool isEmissive(const Material& mat) {
    return mat.emissiveStrength > 0.0f &&
           (mat.emissive.x > 0.0f || mat.emissive.y > 0.0f || mat.emissive.z > 0.0f);
}

} // namespace pt
