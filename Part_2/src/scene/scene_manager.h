#pragma once

#include "scene/scene.h"
#include "scene/device_scene.h"

#include <string>
#include <cstdint>

namespace pt {

class SceneManager {
public:
    void load(const std::string& path);
    void buildAccel();
    void uploadToDevice();

    void updateCamera(const Camera& cam) { deviceScene_.camera = cam; }
    void saveCamera(const Camera& cam);

    // Accessors
    const HostScene&   hostScene()   const { return hostScene_; }
    DeviceScene&       deviceScene()       { return deviceScene_; }
    const DeviceScene& deviceScene() const { return deviceScene_; }
    const Camera&      camera()      const { return hostScene_.camera; }
    const std::string& scenePath()   const { return scenePath_; }
    const std::string& outputPath()  const { return hostScene_.settings.outputFile; }

    // Scene statistics
    uint32_t numVertices()          const { return static_cast<uint32_t>(hostScene_.vertices.size()); }
    uint32_t numFaces()             const { return static_cast<uint32_t>(hostScene_.faces.size()); }
    uint32_t numBvhNodes()          const { return static_cast<uint32_t>(hostScene_.bvh.nodes.size()); }
    uint32_t numMaterials()         const { return static_cast<uint32_t>(hostScene_.materials.size()); }
    uint32_t numEmissiveTriangles() const { return static_cast<uint32_t>(hostScene_.lightSampler.emissiveTriangles.size()); }

    // Settings override (call after load, before uploadToDevice)
    void setSpp(int spp)                 { hostScene_.settings.targetSamples = spp; }
    void setMaxDepth(int depth)          { hostScene_.settings.maxDepth = depth; }
    void setOutput(const std::string& p) { hostScene_.settings.outputFile = p; }
    void setSampler(int type)            { hostScene_.settings.samplerType = type; }

private:
    HostScene   hostScene_;
    DeviceScene deviceScene_;
    std::string scenePath_;
};

} // namespace pt
