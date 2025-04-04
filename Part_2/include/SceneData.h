#pragma once

#include "config.h"
#include "Triangle.h"
#include "Material.h"
#include "BVH.h"
#include "texture_types.h"
#include <cstdint>

BEGIN_NAMESPACE_PT

struct SceneData {
    int width;
    int height;
    uint32_t frameCount;

    Triangle* triangles = nullptr;
    uint32_t numTriangles;

    Material* materials = nullptr;
    uint32_t numMaterials;

    BVHNode* bvhNodes = nullptr;
    uint32_t numBvhNodes; 

    cudaTextureObject_t hdrTex;
};

END_NAMESPACE_PT