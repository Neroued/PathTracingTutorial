#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "io/image_io.h"

#include <cctype>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace pt {

static std::string toLower(const std::string& s) {
    std::string out = s;
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

bool ImageIO::load(Image& image, const std::string& filename) {
    std::string ext;
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        ext = toLower(filename.substr(dotPos));
    } else {
        image = {};
        std::cerr << "ImageIO: unknown file extension: " << filename << std::endl;
        return false;
    }

    int w = 0, h = 0, ch = 0;

    if (ext == ".hdr") {
        float* data = stbi_loadf(filename.c_str(), &w, &h, &ch, 4);
        if (!data) {
            image = {};
            std::cerr << "ImageIO: failed to load: " << filename << std::endl;
            return false;
        }
        image.width   = w;
        image.height  = h;
        image.channel = 4;
        size_t total  = static_cast<size_t>(w) * h * 4;
        image.data.assign(data, data + total);
        stbi_image_free(data);
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i)
            image.data[i * 4 + 3] = 1.0f;
        std::cout << "ImageIO: loaded HDR " << filename << std::endl;
        return true;
    }

    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
        unsigned char* data = stbi_load(filename.c_str(), &w, &h, &ch, 3);
        if (!data) {
            image = {};
            std::cerr << "ImageIO: failed to load: " << filename << std::endl;
            return false;
        }
        image.width   = w;
        image.height  = h;
        image.channel = 4;
        size_t total  = static_cast<size_t>(w) * h;
        image.data.resize(total * 4);
        for (size_t i = 0; i < total; ++i) {
            image.data[i * 4 + 0] = std::pow(data[i * 3 + 0] / 255.0f, 2.2f);
            image.data[i * 4 + 1] = std::pow(data[i * 3 + 1] / 255.0f, 2.2f);
            image.data[i * 4 + 2] = std::pow(data[i * 3 + 2] / 255.0f, 2.2f);
            image.data[i * 4 + 3] = 1.0f;
        }
        stbi_image_free(data);
        std::cout << "ImageIO: loaded " << filename << std::endl;
        return true;
    }

    image = {};
    std::cerr << "ImageIO: unsupported format: " << filename << std::endl;
    return false;
}

bool ImageIO::write(const Image& image, const std::string& filename, bool flipY) {
    std::string ext;
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        ext = toLower(filename.substr(dotPos));
    } else {
        std::cerr << "ImageIO: unknown file extension: " << filename << std::endl;
        return false;
    }

    if (image.data.empty() || image.width <= 0 || image.height <= 0) {
        std::cerr << "ImageIO: invalid image" << std::endl;
        return false;
    }

    size_t totalPixels = static_cast<size_t>(image.width) * image.height;

    std::vector<float> rgb(totalPixels * 3);
    if (image.channel == 3) {
        rgb = image.data;
    } else if (image.channel == 4) {
        for (size_t i = 0; i < totalPixels; ++i) {
            rgb[i * 3 + 0] = image.data[i * 4 + 0];
            rgb[i * 3 + 1] = image.data[i * 4 + 1];
            rgb[i * 3 + 2] = image.data[i * 4 + 2];
        }
    } else {
        std::cerr << "ImageIO: unsupported channel count" << std::endl;
        return false;
    }

    if (flipY) {
        int rowSize = image.width * 3;
        for (int y = 0; y < image.height / 2; ++y) {
            int opp = image.height - 1 - y;
            for (int x = 0; x < rowSize; ++x)
                std::swap(rgb[y * rowSize + x], rgb[opp * rowSize + x]);
        }
    }

    if (ext == ".hdr") {
        return stbi_write_hdr(filename.c_str(), image.width, image.height, 3, rgb.data()) != 0;
    }

    std::vector<unsigned char> ldr(totalPixels * 3);
    for (size_t i = 0; i < totalPixels; ++i) {
        float r = std::clamp(std::pow(rgb[i*3+0], 1.0f/2.2f), 0.0f, 1.0f);
        float g = std::clamp(std::pow(rgb[i*3+1], 1.0f/2.2f), 0.0f, 1.0f);
        float b = std::clamp(std::pow(rgb[i*3+2], 1.0f/2.2f), 0.0f, 1.0f);
        ldr[i*3+0] = static_cast<unsigned char>(r * 255.0f + 0.5f);
        ldr[i*3+1] = static_cast<unsigned char>(g * 255.0f + 0.5f);
        ldr[i*3+2] = static_cast<unsigned char>(b * 255.0f + 0.5f);
    }

    if (ext == ".png")
        return stbi_write_png(filename.c_str(), image.width, image.height, 3, ldr.data(), image.width*3) != 0;
    if (ext == ".jpg" || ext == ".jpeg")
        return stbi_write_jpg(filename.c_str(), image.width, image.height, 3, ldr.data(), 90) != 0;
    if (ext == ".bmp")
        return stbi_write_bmp(filename.c_str(), image.width, image.height, 3, ldr.data()) != 0;
    if (ext == ".tga")
        return stbi_write_tga(filename.c_str(), image.width, image.height, 3, ldr.data()) != 0;

    std::cerr << "ImageIO: unsupported format: " << ext << std::endl;
    return false;
}

} // namespace pt
