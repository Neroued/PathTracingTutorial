#pragma once

#include <vector>
#include <string>

namespace pt {

struct Image {
    int width   = 0;
    int height  = 0;
    int channel = 0;
    std::vector<float> data;

    // Texture sampler wrap modes (GL enum values from glTF)
    int wrapS = 10497;  // GL_REPEAT
    int wrapT = 10497;  // GL_REPEAT
};

class ImageIO {
public:
    static bool load(Image& image, const std::string& filename);
    static bool write(const Image& image, const std::string& filename, bool flipY = true);
};

} // namespace pt
