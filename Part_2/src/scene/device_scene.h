#pragma once

#include "geometry/vertex.h"
#include "materials/material.h"
#include "accel/bvh_common.h"
#include "camera/camera.h"
#include "cuda/cuda_buffer.h"
#include "cuda/cuda_texture.h"
#include "lights/light_sampler.h"
#include "lights/punctual_light.h"

#include <cuda_runtime.h>
#include <vector>

namespace pt {

enum class SamplerType : int { PCG = 0, Sobol = 1 };

struct SceneParams {
    CameraFrame camera;
    int    width;
    int    height;
    int    samplePerFrame;
    int    targetSamples;
    int    maxDepth;
    int    samplerType;   // 0 = PCG, 1 = Sobol
};

struct DeviceSceneView {
    int               width;
    int               height;
    uint32_t          frameCount;

    const Vertex*        vertices;
    uint32_t             numVertices;

    const TriangleFace*  faces;
    uint32_t             numFaces;

    const Material*   materials;
    uint32_t          numMaterials;

    const BVHNode*    bvhNodes;
    uint32_t          numBvhNodes;

    const EmissiveTriangleRef* emissiveTriangles;
    uint32_t          numEmissiveTriangles;

    const float*      triangleLightPmf;
    const AliasEntry* triangleAlias;

    const float*      envPmf;
    const AliasEntry* envAlias;
    uint32_t          numEnvTexels;
    int               envWidth;
    int               envHeight;

    cudaTextureObject_t  hdrTex;

    const cudaTextureObject_t* textures;
    uint32_t                   numTextures;

    const PunctualLight*       punctualLights;
    uint32_t                   numPunctualLights;
};

struct HostScene;

class DeviceScene {
public:
    CudaBuffer<Vertex>        vertices;
    CudaBuffer<TriangleFace>  faces;
    CudaBuffer<Material>      materials;
    CudaBuffer<BVHNode>       bvhNodes;
    CudaBuffer<EmissiveTriangleRef> emissiveTriangles;
    CudaBuffer<float>         triangleLightPmf;
    CudaBuffer<AliasEntry>    triangleAlias;
    CudaBuffer<float>         envPmf;
    CudaBuffer<AliasEntry>    envAlias;
    CudaTexture               hdrTexture;

    std::vector<CudaTexture>       textureStorage;
    CudaBuffer<cudaTextureObject_t> textureHandles;
    CudaBuffer<PunctualLight>      punctualLights;

    Camera                    camera;
    int                       envWidth  = 0;
    int                       envHeight = 0;

    int samplePerFrame = 4;
    int targetSamples  = 100000;
    int maxDepth       = 8;
    int samplerType    = 1;

    uint32_t frameCount = 0;

    void uploadFrom(const HostScene& host);

    SceneParams buildParams() const;

    DeviceSceneView view() const {
        DeviceSceneView v{};
        v.width        = camera.width;
        v.height       = camera.height;
        v.frameCount   = frameCount;
        v.vertices     = vertices.data();
        v.numVertices  = static_cast<uint32_t>(vertices.size());
        v.faces        = faces.data();
        v.numFaces     = static_cast<uint32_t>(faces.size());
        v.materials    = materials.data();
        v.numMaterials = static_cast<uint32_t>(materials.size());
        v.bvhNodes     = bvhNodes.data();
        v.numBvhNodes  = static_cast<uint32_t>(bvhNodes.size());
        v.emissiveTriangles    = emissiveTriangles.data();
        v.numEmissiveTriangles = static_cast<uint32_t>(emissiveTriangles.size());
        v.triangleLightPmf     = triangleLightPmf.data();
        v.triangleAlias        = triangleAlias.data();
        v.envPmf               = envPmf.data();
        v.envAlias             = envAlias.data();
        v.numEnvTexels         = static_cast<uint32_t>(envPmf.size());
        v.envWidth             = envWidth;
        v.envHeight            = envHeight;
        v.hdrTex               = hdrTexture.handle();
        v.textures             = textureHandles.data();
        v.numTextures          = static_cast<uint32_t>(textureHandles.size());
        v.punctualLights       = punctualLights.data();
        v.numPunctualLights    = static_cast<uint32_t>(punctualLights.size());
        return v;
    }
};

} // namespace pt
