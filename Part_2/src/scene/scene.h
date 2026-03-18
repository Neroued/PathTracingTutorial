#pragma once

#include "geometry/vertex.h"
#include "materials/material.h"
#include "accel/bvh_common.h"
#include "camera/camera.h"
#include "io/image_io.h"
#include "lights/light_sampler.h"
#include "lights/punctual_light.h"

#include <vector>
#include <string>

namespace pt {

struct RenderSettings {
    int samplePerFrame = 4;
    int targetSamples  = 100000;
    int maxDepth       = 8;
    int samplerType    = 1;   // 0 = PCG, 1 = Sobol (default: Sobol)
    std::string outputFile = "output.hdr";
};

struct HostScene {
    std::vector<Vertex>        vertices;
    std::vector<TriangleFace>  faces;
    std::vector<Material>      materials;
    std::vector<Image>         textures;
    std::vector<PunctualLight> punctualLights;
    BVH                        bvh;
    Camera                     camera;
    Image                      hdrImage;
    HostLightSampler           lightSampler;
    RenderSettings             settings;
};

} // namespace pt
