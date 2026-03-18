#include "scene/scene_loader.h"
#include "io/mesh_loader.h"
#include "io/image_io.h"
#include "io/gltf_loader.h"
#include "math/vecmath.h"
#include "math/mat4.h"
#include "geometry/vertex.h"
#include "geometry/shapes.h"
#include "materials/material.h"
#include "lights/punctual_light.h"

#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>

using json = nlohmann::json;

namespace pt {

static vec3 toVec3(const json& j) {
    return vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
}

static std::string resolveRelative(const std::string& basePath, const std::string& relPath) {
    namespace fs = std::filesystem;
    fs::path rel(relPath);
    if (rel.is_absolute()) return relPath;
    return (fs::path(basePath).parent_path() / rel).string();
}

// ---- Apply material property overrides from a JSON object ----
static void applyMaterialOverride(const json& p, Material& mat) {
    if (p.contains("baseColor"))          mat.baseColor          = toVec3(p["baseColor"]);
    if (p.contains("metallic"))           mat.metallic           = p["metallic"].get<float>();
    if (p.contains("roughness"))          mat.roughness          = p["roughness"].get<float>();
    if (p.contains("specularTint"))       mat.specularTint       = p["specularTint"].get<float>();
    if (p.contains("anisotropic"))        mat.anisotropic        = p["anisotropic"].get<float>();
    if (p.contains("sheen"))              mat.sheen              = p["sheen"].get<float>();
    if (p.contains("sheenTint"))          mat.sheenTint          = p["sheenTint"].get<float>();
    if (p.contains("clearcoat"))          mat.clearcoat          = p["clearcoat"].get<float>();
    if (p.contains("clearcoatRoughness")) mat.clearcoatRoughness = p["clearcoatRoughness"].get<float>();
    if (p.contains("specTrans"))          mat.specTrans          = p["specTrans"].get<float>();
    if (p.contains("ior"))                mat.ior                = p["ior"].get<float>();
    if (p.contains("subsurface"))         mat.subsurface         = p["subsurface"].get<float>();
    if (p.contains("emissive"))           mat.emissive           = toVec3(p["emissive"]);
    if (p.contains("emissiveStrength"))   mat.emissiveStrength   = p["emissiveStrength"].get<float>();
    if (p.contains("alphaCutoff"))        mat.alphaCutoff        = p["alphaCutoff"].get<float>();
    if (p.contains("alphaMode"))          mat.alphaMode          = p["alphaMode"].get<int>();
    if (p.contains("doubleSided"))        mat.doubleSided        = p["doubleSided"].get<bool>() ? 1 : 0;
}

// ---- Parse transform block ----
static mat4 parseTransform(const json& obj) {
    if (!obj.contains("transform")) return mat4();
    auto& t = obj["transform"];
    vec3 translate(0, 0, 0);
    vec3 axis(0, 1, 0);
    float angle = 0.0f;
    float scaleVal = 1.0f;

    if (t.contains("translate")) translate = toVec3(t["translate"]);
    if (t.contains("rotate")) {
        axis  = toVec3(t["rotate"]["axis"]);
        angle = t["rotate"]["angle"].get<float>();
    }
    if (t.contains("scale")) scaleVal = t["scale"].get<float>();

    return mat4::translation(translate)
         * mat4::rotation(angle * Pi / 180.0f, axis)
         * mat4::scaling(scaleVal, scaleVal, scaleVal);
}

static void parseCamera(const json& root, Camera& camera, bool& cameraSet) {
    if (!root.contains("camera")) return;
    auto& cam = root["camera"];
    if (cam.contains("position"))      camera.position      = toVec3(cam["position"]);
    if (cam.contains("lookAt"))        camera.lookAt        = toVec3(cam["lookAt"]);
    if (cam.contains("up"))            camera.up            = toVec3(cam["up"]);
    if (cam.contains("fov"))           camera.fov           = cam["fov"].get<float>();
    if (cam.contains("aperture"))      camera.aperture      = cam["aperture"].get<float>();
    if (cam.contains("focalDistance")) camera.focalDistance  = cam["focalDistance"].get<float>();
    if (cam.contains("resolution")) {
        camera.width  = cam["resolution"][0].get<int>();
        camera.height = cam["resolution"][1].get<int>();
    }
    cameraSet = true;
}

static void parseRenderSettings(const json& root, RenderSettings& s) {
    if (!root.contains("render")) return;
    auto& r = root["render"];
    if (r.contains("spp"))            s.targetSamples  = r["spp"].get<int>();
    if (r.contains("samplePerFrame")) s.samplePerFrame = r["samplePerFrame"].get<int>();
    if (r.contains("maxDepth"))       s.maxDepth       = r["maxDepth"].get<int>();
    if (r.contains("output"))         s.outputFile     = r["output"].get<std::string>();
    if (r.contains("sampler")) {
        std::string sampler = r["sampler"].get<std::string>();
        if (sampler == "pcg")        s.samplerType = 0;
        else if (sampler == "sobol") s.samplerType = 1;
    }
}

static void parseEnvironment(const json& root, const std::string& jsonPath, Image& hdrImage) {
    if (!root.contains("environment")) return;
    if (!root["environment"].contains("file")) return;
    std::string hdrPath = resolveRelative(jsonPath, root["environment"]["file"].get<std::string>());
    ImageIO::load(hdrImage, hdrPath);
}

static void parseMaterials(const json& root,
                           std::vector<Material>& materials,
                           std::map<std::string, int>& matIndex) {
    if (!root.contains("materials")) return;
    for (auto& [name, p] : root["materials"].items()) {
        Material mat;
        applyMaterialOverride(p, mat);
        matIndex[name] = static_cast<int>(materials.size());
        materials.push_back(mat);
    }
}

static void parseLights(const json& root, std::vector<PunctualLight>& lights) {
    if (!root.contains("lights")) return;
    for (auto& jl : root["lights"]) {
        PunctualLight pl;
        std::string type = jl.value("type", std::string("point"));
        if (type == "directional") pl.type = LightType::Directional;
        else if (type == "spot")   pl.type = LightType::Spot;
        else                       pl.type = LightType::Point;

        if (jl.contains("position"))  pl.position  = toVec3(jl["position"]);
        if (jl.contains("direction")) pl.direction  = normalize(toVec3(jl["direction"]));
        if (jl.contains("color"))     pl.color      = toVec3(jl["color"]);
        if (jl.contains("intensity")) pl.intensity  = jl["intensity"].get<float>();
        if (jl.contains("range"))     pl.range      = jl["range"].get<float>();
        if (jl.contains("innerCone")) pl.innerCone  = std::cos(jl["innerCone"].get<float>());
        if (jl.contains("outerCone")) pl.outerCone  = std::cos(jl["outerCone"].get<float>());
        lights.push_back(pl);
    }
}

static void parseObjects(const json& root,
                          const std::string& jsonPath,
                          const std::map<std::string, int>& matIndex,
                          HostScene& scene) {
    if (!root.contains("objects")) return;

    auto resolveMat = [&](const json& j) -> int {
        if (j.is_string()) {
            auto it = matIndex.find(j.get<std::string>());
            return it != matIndex.end() ? it->second : 0;
        }
        return j.get<int>();
    };

    for (auto& obj : root["objects"]) {
        std::string type = obj.value("type", std::string(""));
        std::string name = obj.value("name", std::string(""));

        if (type == "gltf") {
            std::string gltfFile = resolveRelative(jsonPath, obj["file"].get<std::string>());
            mat4 xform = parseTransform(obj);

            int materialStart = static_cast<int>(scene.materials.size());

            std::vector<std::string> matNames;
            GltfLoadOptions opts;
            opts.transform        = xform;
            opts.loadCamera       = false;
            opts.autoFitCamera    = false;
            opts.addDefaultLights = false;
            opts.loadEnvironment  = false;
            opts.materialNames    = &matNames;

            if (!name.empty())
                std::cout << "SceneLoader: loading gltf object '" << name << "': " << gltfFile << std::endl;

            GltfSceneLoader::load(gltfFile, scene, opts);

            // Apply materialOverrides by matching GLTF material names
            if (obj.contains("materialOverrides")) {
                for (auto& [matName, overrides] : obj["materialOverrides"].items()) {
                    for (size_t i = 0; i < matNames.size(); ++i) {
                        if (matNames[i] == matName) {
                            int idx = materialStart + static_cast<int>(i);
                            if (idx < static_cast<int>(scene.materials.size()))
                                applyMaterialOverride(overrides, scene.materials[idx]);
                        }
                    }
                }
            }
        }
        else if (type == "quad") {
            auto verts = obj["vertices"];
            vec3 a = toVec3(verts[0]), b = toVec3(verts[1]);
            vec3 c = toVec3(verts[2]), d = toVec3(verts[3]);
            shapes::buildQuad(a, b, c, d, resolveMat(obj["material"]),
                              scene.vertices, scene.faces);
        }
        else if (type == "mesh") {
            std::string meshFile = resolveRelative(jsonPath, obj["file"].get<std::string>());
            int matId = resolveMat(obj["material"]);
            mat4 xform = parseTransform(obj);
            MeshLoader::loadObj(meshFile, matId, xform, scene.vertices, scene.faces);
        }
    }
}

// Auto-fit camera from geometry bounding box
static void autoFitCamera(const std::vector<Vertex>& vertices, Camera& camera) {
    if (vertices.empty()) return;
    vec3 bmin( 1e30f,  1e30f,  1e30f);
    vec3 bmax(-1e30f, -1e30f, -1e30f);
    for (const auto& v : vertices) {
        bmin.x = std::fmin(bmin.x, v.position.x);
        bmin.y = std::fmin(bmin.y, v.position.y);
        bmin.z = std::fmin(bmin.z, v.position.z);
        bmax.x = std::fmax(bmax.x, v.position.x);
        bmax.y = std::fmax(bmax.y, v.position.y);
        bmax.z = std::fmax(bmax.z, v.position.z);
    }
    vec3 center = (bmin + bmax) * 0.5f;
    float radius = (bmax - bmin).length() * 0.5f;
    if (radius < 1e-5f) radius = 1.0f;

    float fovRad = camera.fov * Pi / 180.0f;
    float dist   = radius / std::tan(fovRad * 0.5f) * 1.2f;

    camera.position = center + vec3(0.0f, radius * 0.3f, dist);
    camera.lookAt   = center;
    camera.up       = vec3(0, 1, 0);

    std::cout << "SceneLoader: auto-fit camera  center=("
              << center.x << ", " << center.y << ", " << center.z
              << ")  dist=" << dist << std::endl;
}

bool SceneLoader::loadFromJson(const std::string& jsonPath, HostScene& scene) {
    std::ifstream f(jsonPath);
    if (!f.is_open()) {
        std::cerr << "SceneLoader: cannot open " << jsonPath << std::endl;
        return false;
    }

    json root;
    try {
        f >> root;
    } catch (const json::parse_error& e) {
        std::cerr << "SceneLoader: JSON parse error: " << e.what() << std::endl;
        return false;
    }

    std::map<std::string, int> matIndex;
    bool cameraSet = false;

    parseCamera(root, scene.camera, cameraSet);
    parseRenderSettings(root, scene.settings);
    parseEnvironment(root, jsonPath, scene.hdrImage);
    parseMaterials(root, scene.materials, matIndex);
    parseLights(root, scene.punctualLights);
    parseObjects(root, jsonPath, matIndex, scene);

    if (scene.materials.empty())
        scene.materials.push_back(Material{});

    if (!cameraSet)
        autoFitCamera(scene.vertices, scene.camera);

    std::cout << "SceneLoader: " << scene.faces.size() << " triangles, "
              << scene.vertices.size() << " vertices, "
              << scene.materials.size() << " materials, "
              << scene.punctualLights.size() << " lights" << std::endl;
    return true;
}

bool SceneLoader::load(const std::string& path, HostScene& scene) {
    namespace fs = std::filesystem;
    std::string ext = fs::path(path).extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (ext == ".json") {
        return loadFromJson(path, scene);
    }
    if (ext == ".gltf" || ext == ".glb") {
        return GltfSceneLoader::load(path, scene);
    }

    std::cerr << "SceneLoader: unknown scene format: " << ext << std::endl;
    return false;
}

bool SceneLoader::saveCamera(const std::string& jsonPath, const Camera& camera) {
    std::ifstream fin(jsonPath);
    if (!fin.is_open()) {
        std::cerr << "SceneLoader::saveCamera: cannot open " << jsonPath << std::endl;
        return false;
    }

    json root;
    try {
        fin >> root;
    } catch (const json::parse_error& e) {
        std::cerr << "SceneLoader::saveCamera: parse error: " << e.what() << std::endl;
        return false;
    }
    fin.close();

    root["camera"]["position"]      = {camera.position.x, camera.position.y, camera.position.z};
    root["camera"]["lookAt"]        = {camera.lookAt.x, camera.lookAt.y, camera.lookAt.z};
    root["camera"]["up"]            = {camera.up.x, camera.up.y, camera.up.z};
    root["camera"]["fov"]           = camera.fov;
    root["camera"]["aperture"]      = camera.aperture;
    root["camera"]["focalDistance"]  = camera.focalDistance;
    root["camera"]["resolution"]    = {camera.width, camera.height};

    std::ofstream fout(jsonPath);
    if (!fout.is_open()) {
        std::cerr << "SceneLoader::saveCamera: cannot write " << jsonPath << std::endl;
        return false;
    }
    fout << root.dump(2) << std::endl;
    std::cout << "SceneLoader: saved camera to " << jsonPath << std::endl;
    return true;
}

} // namespace pt
