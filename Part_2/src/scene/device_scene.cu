#include "scene/device_scene.h"
#include "scene/scene.h"

#include <iostream>

namespace pt {

void DeviceScene::uploadFrom(const HostScene& host) {
    vertices.upload(host.vertices);
    faces.upload(host.faces);
    materials.upload(host.materials);
    bvhNodes.upload(host.bvh.nodes);
    emissiveTriangles.upload(host.lightSampler.emissiveTriangles);
    triangleLightPmf.upload(host.lightSampler.triangleSelectionPmf);
    triangleAlias.upload(host.lightSampler.triangleAlias);
    envPmf.upload(host.lightSampler.envPmf);
    envAlias.upload(host.lightSampler.envAlias);

    if (host.hdrImage.channel == 4) {
        hdrTexture = CudaTexture(host.hdrImage);
        std::cout << "HDR texture: " << host.hdrImage.width << "x" << host.hdrImage.height
                  << " handle=" << hdrTexture.handle() << std::endl;
    } else {
        hdrTexture = CudaTexture();
        std::cerr << "WARNING: No HDR environment loaded (channel="
                  << host.hdrImage.channel << ")" << std::endl;
    }

    // Upload material textures
    textureStorage.clear();
    textureStorage.reserve(host.textures.size());
    std::vector<cudaTextureObject_t> handles;
    handles.reserve(host.textures.size());
    for (const auto& img : host.textures) {
        textureStorage.emplace_back(img);
        handles.push_back(textureStorage.back().handle());
    }
    textureHandles.upload(handles);
    if (!handles.empty()) {
        std::cout << "Material textures: " << handles.size() << " uploaded" << std::endl;
    }

    // Upload punctual lights
    punctualLights.upload(host.punctualLights);
    if (!host.punctualLights.empty()) {
        std::cout << "Punctual lights: " << host.punctualLights.size() << " uploaded" << std::endl;
    }

    camera         = host.camera;
    envWidth       = host.lightSampler.envWidth;
    envHeight      = host.lightSampler.envHeight;
    samplePerFrame = host.settings.samplePerFrame;
    targetSamples  = host.settings.targetSamples;
    maxDepth       = host.settings.maxDepth;
    samplerType    = host.settings.samplerType;
    frameCount     = 0;
}

SceneParams DeviceScene::buildParams() const {
    SceneParams p{};
    p.camera         = buildCameraFrame(camera);
    p.width          = camera.width;
    p.height         = camera.height;
    p.samplePerFrame = samplePerFrame;
    p.targetSamples  = targetSamples;
    p.maxDepth       = maxDepth;
    p.samplerType    = samplerType;
    return p;
}

} // namespace pt
