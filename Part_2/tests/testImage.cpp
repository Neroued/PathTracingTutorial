#include "ImageLoader.h"
#include <iostream>

using namespace pt;

int main() {
    Image img;
    bool ok = ImageLoader::load(img, "E:\\code\\c++\\PathTracingTutorial\\Part_2\\models\\circus_arena_4k.hdr");

    if (!ok) {
        std::cout << "failed" << std::endl;
    } else {
        std::cout << "width: " << img.width << " height: " << img.height << " channel: " << img.channel << " total floats: " << img.data.size()
                  << std::endl;
    }

    ok = ImageLoader::write(img, "test.png");
    if (!ok) { std::cout << "failed save" << std::endl; }

    return 0;
}