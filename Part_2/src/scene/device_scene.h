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

struct DeviceLightView {
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

    const PunctualLight*       punctualLights;
    uint32_t                   numPunctualLights;
};

struct DeviceSceneView {
    uint32_t          frameCount;

    const Vertex*        vertices;
    uint32_t             numVertices;

    const TriangleFace*  faces;
    uint32_t             numFaces;

    const Material*   materials;
    uint32_t          numMaterials;

    const BVHNode*    bvhNodes;
    uint32_t          numBvhNodes;

    const cudaTextureObject_t* textures;
    uint32_t                   numTextures;

    DeviceLightView   lights;
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
        v.frameCount   = frameCount;
        v.vertices     = vertices.data();
        v.numVertices  = static_cast<uint32_t>(vertices.size());
        v.faces        = faces.data();
        v.numFaces     = static_cast<uint32_t>(faces.size());
        v.materials    = materials.data();
        v.numMaterials = static_cast<uint32_t>(materials.size());
        v.bvhNodes     = bvhNodes.data();
        v.numBvhNodes  = static_cast<uint32_t>(bvhNodes.size());
        v.textures             = textureHandles.data();
        v.numTextures          = static_cast<uint32_t>(textureHandles.size());
        v.lights.emissiveTriangles    = emissiveTriangles.data();
        v.lights.numEmissiveTriangles = static_cast<uint32_t>(emissiveTriangles.size());
        v.lights.triangleLightPmf    = triangleLightPmf.data();
        v.lights.triangleAlias       = triangleAlias.data();
        v.lights.envPmf              = envPmf.data();
        v.lights.envAlias            = envAlias.data();
        v.lights.numEnvTexels        = static_cast<uint32_t>(envPmf.size());
        v.lights.envWidth            = envWidth;
        v.lights.envHeight           = envHeight;
        v.lights.hdrTex              = hdrTexture.handle();
        v.lights.punctualLights      = punctualLights.data();
        v.lights.numPunctualLights   = static_cast<uint32_t>(punctualLights.size());
        return v;
    }
};

} // namespace pt
