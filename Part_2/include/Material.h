#pragma once

#include "config.h"
#include "vec3.h"

BEGIN_NAMESPACE_PT

struct Material {
    vec3 baseColor         = {0.0f, 0.0f, 0.0f};
    float specularRate     = 0.0f;
    float roughness        = 1.0f;
    float refractRate      = 0.0f;
    float refractAngle     = 1.0f;
    float refractRoughness = 0.0f;
    float emissive         = 0.0f;
};

END_NAMESPACE_PT