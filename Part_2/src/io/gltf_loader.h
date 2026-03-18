#pragma once

#include "math/mat4.h"

#include <string>
#include <vector>

namespace pt {

struct HostScene;

struct GltfLoadOptions {
    mat4 transform        = mat4();
    bool loadCamera       = true;      // use embedded glTF camera
    bool autoFitCamera    = true;      // fallback: fit camera to bounding box
    bool addDefaultLights = true;      // fallback: add default lights if none found
    bool loadEnvironment  = true;      // fallback: search for default HDR
    std::vector<std::string>* materialNames = nullptr;
};

class GltfSceneLoader {
public:
    static bool load(const std::string& path, HostScene& scene,
                     const GltfLoadOptions& opts = {});
};

} // namespace pt
