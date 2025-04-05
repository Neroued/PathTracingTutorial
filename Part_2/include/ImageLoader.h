#pragma once

#include "config.h"
#include <vector>
#include <string>

BEGIN_NAMESPACE_PT

struct Image {
    int width;
    int height;
    int channel;
    std::vector<float> data;
};

class ImageLoader {
public:
    // 加载图像默认为 4 个通道，A 通道固定为 1.0f
    static bool load(Image& image, const std::string& filename);

    // 根据扩展名保存为不同格式，支持 hdr
    static bool write(const Image& image, const std::string& filename, bool flipY = true);
};

END_NAMESPACE_PT