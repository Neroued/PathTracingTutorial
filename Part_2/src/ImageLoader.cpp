#include "config.h"
#include "ImageLoader.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <string>
#include <vector>
#include <cctype>
#include <iostream>
#include <algorithm>
#include <cstddef>

BEGIN_NAMESPACE_PT

static std::string toLower(const std::string& s) {
    std::string out = s;
    for (char& c : out) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
    return out;
}

bool ImageLoader::load(Image& image, const std::string& filename) {
    // 提取文件扩展名
    std::string ext;
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        ext = toLower(filename.substr(dotPos));
    } else {
        image.data.clear();
        image.width = image.height = image.channel = 0;
        std::cerr << "ImageLoader: Unknown file: " << filename << std::endl;
        return false;
    }

    int width = 0, height = 0, channels_in_file = 0;
    if (ext == ".hdr") {
        // HDR 使用 stbi_loadf 保存浮点，本身数据就在线性空间
        float* data = stbi_loadf(filename.c_str(), &width, &height, &channels_in_file, 4);
        if (!data) {
            image.data.clear();
            image.width = image.height = image.channel = 0;
            std::cerr << "ImageLoader: Failed to load file: " << filename << std::endl;
            return false;
        }
        image.width        = width;
        image.height       = height;
        image.channel      = 4;
        size_t totalFloats = static_cast<size_t>(width) * height * 4;
        image.data.assign(data, data + totalFloats);
        stbi_image_free(data);
        // 确保 alpha 通道数据为 1.0f
        for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) { image.data[i * 4 + 3] = 1.0f; }
        return true;
    } else if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
        // JPG 和 PNG 使用 stbi_load 加载 8 位数据
        unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels_in_file, 3);
        if (!data) {
            image.data.clear();
            image.width = image.height = image.channel = 0;
            std::cerr << "ImageLoader: Failed to load file: " << filename << std::endl;
            return false;
        }
        image.width        = width;
        image.height       = height;
        image.channel      = 4;
        size_t totalPixels = static_cast<size_t>(width) * height;
        image.data.resize(totalPixels * image.channel);
        // 对每个像素执行 gamma 解码，将 sRGB 数据转换为线性空间, 同时补充第四个通道
        for (size_t i = 0; i < totalPixels; ++i) {
            float r               = std::pow(data[i * 3 + 0] / 255.0f, 2.2f);
            float g               = std::pow(data[i * 3 + 1] / 255.0f, 2.2f);
            float b               = std::pow(data[i * 3 + 2] / 255.0f, 2.2f);
            image.data[i * 4 + 0] = r;
            image.data[i * 4 + 1] = g;
            image.data[i * 4 + 2] = b;
            image.data[i * 4 + 3] = 1.0f;
        }
        stbi_image_free(data);
        return true;
    } else {
        image.data.clear();
        image.width = image.height = image.channel = 0;
        std::cerr << "ImageLoader: Unsupport file: " << filename << std::endl;
        return false;
    }
}

bool ImageLoader::write(const Image& image, const std::string& filename) {
    // 提取文件扩展名
    std::string ext;
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        ext = toLower(filename.substr(dotPos));
    } else {
        std::cerr << "ImageLoader: Unknown file: " << filename << std::endl;
        return false;
    }

    // 判断 image 是否合法
    if (image.data.empty() || image.width <= 0 || image.height <= 0 ||
        image.data.size() < static_cast<size_t>(image.width) * image.height * image.channel) {
        std::cerr << "ImageLoader: Invalid image" << std::endl;
        return false;
    }

    std::vector<float> rgbData;
    size_t totalPixels = static_cast<size_t>(image.width) * image.height;
    if (image.channel == 3) {
        rgbData = image.data;
    } else if (image.channel == 4) {
        rgbData.resize(totalPixels * 3);
        for (size_t i = 0; i < totalPixels; ++i) {
            rgbData[i * 3 + 0] = image.data[i * 4 + 0];
            rgbData[i * 3 + 1] = image.data[i * 4 + 1];
            rgbData[i * 3 + 2] = image.data[i * 4 + 2];
        }
    } else {
        std::cerr << "ImageLoader: Invalid image" << std::endl;
        return false;
    }

    if (ext == ".hdr") {
        int result = stbi_write_hdr(filename.c_str(), image.width, image.height, 3, rgbData.data());
        return result != 0;
    } else {
        // 将内容转换为 8 位数据
        std::vector<unsigned char> outData(totalPixels * 3);
        for (size_t i = 0; i < totalPixels; ++i) {
            // 伽马矫正
            const float invGamma = 1.0f / 2.2f;
            float r              = std::pow(rgbData[i * 3 + 0], invGamma);
            float g              = std::pow(rgbData[i * 3 + 1], invGamma);
            float b              = std::pow(rgbData[i * 3 + 2], invGamma);
            // 限制数值范围
            r = std::clamp(r, 0.0f, 1.0f);
            g = std::clamp(g, 0.0f, 1.0f);
            b = std::clamp(b, 0.0f, 1.0f);

            outData[i * 3 + 0] = static_cast<unsigned char>(r * 255.0f + 0.5f);
            outData[i * 3 + 1] = static_cast<unsigned char>(g * 255.0f + 0.5f);
            outData[i * 3 + 2] = static_cast<unsigned char>(b * 255.0f + 0.5f);
        }
        // 根据扩展名选择
        int result = 0;
        if (ext == ".png") {
            result = stbi_write_png(filename.c_str(), image.width, image.height, 3, outData.data(), image.width * 3);
        } else if (ext == ".jpg" || ext == "jpeg") {
            result = stbi_write_jpg(filename.c_str(), image.width, image.height, 3, outData.data(), 90);
        } else if (ext == ".bmp") {
            result = stbi_write_bmp(filename.c_str(), image.width, image.height, 3, outData.data());
        } else if (ext == ".tga") {
            result = stbi_write_tga(filename.c_str(), image.width, image.height, 3, outData.data());
        } else {
            std::cerr << "Unsupport format" << std::endl;
        }
        return result != 0;
    }
}

END_NAMESPACE_PT