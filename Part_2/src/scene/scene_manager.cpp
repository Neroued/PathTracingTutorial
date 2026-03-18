#include "scene/scene_manager.h"
#include "scene/scene_loader.h"
#include "accel/bvh_builder.h"
#include "lights/light_sampler.h"

#include <iostream>
#include <filesystem>

namespace pt {

void SceneManager::load(const std::string& path) {
    scenePath_ = path;
    SceneLoader::load(path, hostScene_);
}

void SceneManager::saveCamera(const Camera& cam) {
    if (scenePath_.empty()) return;
    namespace fs = std::filesystem;
    std::string ext = fs::path(scenePath_).extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext != ".json") {
        std::cerr << "SceneManager::saveCamera: can only save to .json scenes" << std::endl;
        return;
    }
    SceneLoader::saveCamera(scenePath_, cam);
}

void SceneManager::buildAccel() {
    std::cout << "Building BVH..." << std::endl;
    BVHBuilder builder;
    hostScene_.bvh = builder.build(hostScene_.vertices, hostScene_.faces, 0,
                                   static_cast<uint32_t>(hostScene_.faces.size()));
    buildEmissiveTriangleSampler(hostScene_.vertices, hostScene_.faces,
                                 hostScene_.materials, hostScene_.lightSampler);
    buildEnvironmentSampler(hostScene_.hdrImage, hostScene_.lightSampler);
    std::cout << "BVH: " << hostScene_.bvh.nodes.size() << " nodes, "
              << hostScene_.faces.size() << " faces, "
              << hostScene_.vertices.size() << " vertices, "
              << hostScene_.lightSampler.emissiveTriangles.size() << " emissive triangles"
              << std::endl;
}

void SceneManager::uploadToDevice() {
    deviceScene_.uploadFrom(hostScene_);
    std::cout << "Scene uploaded to GPU (sampler: "
              << (hostScene_.settings.samplerType == 1 ? "sobol" : "pcg") << ")" << std::endl;
}

} // namespace pt
