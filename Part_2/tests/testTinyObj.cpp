#include <iostream>
#include "tiny_obj_loader.h"

std::string inputfile = "E:\\code\\c++\\PathTracingTutorial\\Part_2\\models\\teapot.obj";

void printMaterial(const tinyobj::material_t& mat);

int main() {
    
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "";

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) { std::cerr << "TinyObjReader: " << reader.Error(); }
        exit(1);
    }

    if (!reader.Warning().empty()) { std::cout << "TinyObjReader: " << reader.Warning(); }

    auto& attrib    = reader.GetAttrib();
    auto& shapes    = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    std::cout << "Get " << materials.size() << " materials" << std::endl;
    
    for (const auto& mat : materials) {
        printMaterial(mat);
    }
    return 0;
}

// 辅助函数，将 texture_type_t 转换为字符串
std::string textureTypeToString(tinyobj::texture_type_t type) {
    switch (type) {
    case tinyobj::TEXTURE_TYPE_NONE:
        return "";
    case tinyobj::TEXTURE_TYPE_SPHERE:
        return "SPHERE";
    case tinyobj::TEXTURE_TYPE_CUBE_TOP:
        return "CUBE_TOP";
    case tinyobj::TEXTURE_TYPE_CUBE_BOTTOM:
        return "CUBE_BOTTOM";
    case tinyobj::TEXTURE_TYPE_CUBE_FRONT:
        return "CUBE_FRONT";
    case tinyobj::TEXTURE_TYPE_CUBE_BACK:
        return "CUBE_BACK";
    case tinyobj::TEXTURE_TYPE_CUBE_LEFT:
        return "CUBE_LEFT";
    case tinyobj::TEXTURE_TYPE_CUBE_RIGHT:
        return "CUBE_RIGHT";
    default:
        return "UNKNOWN";
    }
}

// 重载输出运算符，用于输出 texture_option_t 的所有内容
std::ostream& operator<<(std::ostream& os, const tinyobj::texture_option_t& texopt) {
    if (textureTypeToString(texopt.type).empty()) return os;
    os << "{ type: " << textureTypeToString(texopt.type) << ", sharpness: " << texopt.sharpness << ", brightness: " << texopt.brightness
       << ", contrast: " << texopt.contrast << ", origin_offset: (" << texopt.origin_offset[0] << ", " << texopt.origin_offset[1] << ", "
       << texopt.origin_offset[2] << ")"
       << ", scale: (" << texopt.scale[0] << ", " << texopt.scale[1] << ", " << texopt.scale[2] << ")"
       << ", turbulence: (" << texopt.turbulence[0] << ", " << texopt.turbulence[1] << ", " << texopt.turbulence[2] << ")"
       << ", texture_resolution: " << texopt.texture_resolution << ", clamp: " << (texopt.clamp ? "true" : "false") << ", imfchan: " << texopt.imfchan
       << ", blendu: " << (texopt.blendu ? "true" : "false") << ", blendv: " << (texopt.blendv ? "true" : "false")
       << ", bump_multiplier: " << texopt.bump_multiplier;
    if (!texopt.colorspace.empty()) { os << ", colorspace: " << texopt.colorspace; }
    os << " }";
    return os;
}

// 使用 std::cout 完整输出 material_t 的内容
void printMaterial(const tinyobj::material_t& mat) {
    std::cout << "----- Material -----" << std::endl;

    std::cout << "Name: " << (mat.name.empty() ? "N/A" : mat.name) << std::endl;

    std::cout << "Ambient: (" << mat.ambient[0] << ", " << mat.ambient[1] << ", " << mat.ambient[2] << ")" << std::endl;
    std::cout << "Diffuse: (" << mat.diffuse[0] << ", " << mat.diffuse[1] << ", " << mat.diffuse[2] << ")" << std::endl;
    std::cout << "Specular: (" << mat.specular[0] << ", " << mat.specular[1] << ", " << mat.specular[2] << ")" << std::endl;
    std::cout << "Transmittance: (" << mat.transmittance[0] << ", " << mat.transmittance[1] << ", " << mat.transmittance[2] << ")" << std::endl;
    std::cout << "Emission: (" << mat.emission[0] << ", " << mat.emission[1] << ", " << mat.emission[2] << ")" << std::endl;

    std::cout << "Shininess: " << mat.shininess << std::endl;
    std::cout << "IOR: " << mat.ior << std::endl;
    std::cout << "Dissolve: " << mat.dissolve << std::endl;
    std::cout << "Illum: " << mat.illum << std::endl;

    std::cout << "Dummy: " << mat.dummy << std::endl;

    if (!mat.ambient_texname.empty()) std::cout << "Ambient Texname: " << mat.ambient_texname << std::endl;
    if (!mat.diffuse_texname.empty()) std::cout << "Diffuse Texname: " << mat.diffuse_texname << std::endl;
    if (!mat.specular_texname.empty()) std::cout << "Specular Texname: " << mat.specular_texname << std::endl;
    if (!mat.specular_highlight_texname.empty()) std::cout << "Specular Highlight Texname: " << mat.specular_highlight_texname << std::endl;
    if (!mat.bump_texname.empty()) std::cout << "Bump Texname: " << mat.bump_texname << std::endl;
    if (!mat.displacement_texname.empty()) std::cout << "Displacement Texname: " << mat.displacement_texname << std::endl;
    if (!mat.alpha_texname.empty()) std::cout << "Alpha Texname: " << mat.alpha_texname << std::endl;
    if (!mat.reflection_texname.empty()) std::cout << "Reflection Texname: " << mat.reflection_texname << std::endl;

    std::cout << "Ambient Texopt: " << mat.ambient_texopt << std::endl;
    std::cout << "Diffuse Texopt: " << mat.diffuse_texopt << std::endl;
    std::cout << "Specular Texopt: " << mat.specular_texopt << std::endl;
    std::cout << "Specular Highlight Texopt: " << mat.specular_highlight_texopt << std::endl;
    std::cout << "Bump Texopt: " << mat.bump_texopt << std::endl;
    std::cout << "Displacement Texopt: " << mat.displacement_texopt << std::endl;
    std::cout << "Alpha Texopt: " << mat.alpha_texopt << std::endl;
    std::cout << "Reflection Texopt: " << mat.reflection_texopt << std::endl;

    std::cout << "----- PBR Extension -----" << std::endl;
    std::cout << "Roughness: " << mat.roughness << std::endl;
    std::cout << "Metallic: " << mat.metallic << std::endl;
    std::cout << "Sheen: " << mat.sheen << std::endl;
    std::cout << "Clearcoat Thickness: " << mat.clearcoat_thickness << std::endl;
    std::cout << "Clearcoat Roughness: " << mat.clearcoat_roughness << std::endl;
    std::cout << "Anisotropy: " << mat.anisotropy << std::endl;
    std::cout << "Anisotropy Rotation: " << mat.anisotropy_rotation << std::endl;

    if (!mat.roughness_texname.empty()) std::cout << "Roughness Texname: " << mat.roughness_texname << std::endl;
    if (!mat.metallic_texname.empty()) std::cout << "Metallic Texname: " << mat.metallic_texname << std::endl;
    if (!mat.sheen_texname.empty()) std::cout << "Sheen Texname: " << mat.sheen_texname << std::endl;
    if (!mat.emissive_texname.empty()) std::cout << "Emissive Texname: " << mat.emissive_texname << std::endl;
    if (!mat.normal_texname.empty()) std::cout << "Normal Texname: " << mat.normal_texname << std::endl;

    std::cout << "Roughness Texopt: " << mat.roughness_texopt << std::endl;
    std::cout << "Metallic Texopt: " << mat.metallic_texopt << std::endl;
    std::cout << "Sheen Texopt: " << mat.sheen_texopt << std::endl;
    std::cout << "Emissive Texopt: " << mat.emissive_texopt << std::endl;
    std::cout << "Normal Texopt: " << mat.normal_texopt << std::endl;

    if (!mat.unknown_parameter.empty()) {
        std::cout << "Unknown Parameters:" << std::endl;
        for (const auto& kv : mat.unknown_parameter) { std::cout << "  " << kv.first << " = " << kv.second << std::endl; }
    }

    std::cout << "--------------------" << std::endl;
}