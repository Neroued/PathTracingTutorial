#pragma once

#include "scene/scene.h"
#include <string>

namespace pt {

class SceneLoader {
public:
    static bool load(const std::string& path, HostScene& scene);
    static bool loadFromJson(const std::string& jsonPath, HostScene& scene);
    static bool saveCamera(const std::string& jsonPath, const Camera& camera);
};

} // namespace pt
