#include <nlohmann/json.hpp>
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_INCLUDE_JSON
#include "tiny_gltf.h"

#include "io/gltf_loader.h"
#include "scene/scene.h"
#include "math/mat4.h"
#include "io/image_io.h"

#include "stb/stb_image.h"

#include <iostream>
#include <filesystem>
#include <cmath>

namespace {

// Custom image loader callback for tinygltf (uses stb_image linked from stb_impl.cpp)
bool LoadImageData(tinygltf::Image* image, const int image_idx, std::string* err,
                   std::string* warn, int req_width, int req_height,
                   const unsigned char* bytes, int size, void* user_data) {
    (void)image_idx; (void)warn; (void)req_width; (void)req_height; (void)user_data;
    int w, h, comp;
    unsigned char* data = stbi_load_from_memory(bytes, size, &w, &h, &comp, 4);
    if (!data) {
        if (err) *err = "stbi_load_from_memory failed";
        return false;
    }
    image->width     = w;
    image->height    = h;
    image->component = 4;
    image->bits      = 8;
    image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
    image->image.assign(data, data + static_cast<size_t>(w) * h * 4);
    stbi_image_free(data);
    return true;
}

bool WriteImageData(const std::string*, const std::string*, const tinygltf::Image*,
                    bool, const tinygltf::FsCallbacks*, const tinygltf::URICallbacks*,
                    std::string*, void*) {
    return true;
}

} // anonymous namespace

namespace pt {

// ---- Helper: extract mat4 from a glTF node ----
static mat4 nodeTransform(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        float m[16];
        for (int i = 0; i < 16; ++i) m[i] = static_cast<float>(node.matrix[i]);
        // glTF stores column-major, our mat4 is row-major
        return mat4(m[0], m[4], m[8],  m[12],
                    m[1], m[5], m[9],  m[13],
                    m[2], m[6], m[10], m[14],
                    m[3], m[7], m[11], m[15]);
    }

    mat4 T, R, S;
    if (node.translation.size() == 3)
        T = mat4::translation(static_cast<float>(node.translation[0]),
                              static_cast<float>(node.translation[1]),
                              static_cast<float>(node.translation[2]));

    if (node.rotation.size() == 4) {
        float qx = static_cast<float>(node.rotation[0]);
        float qy = static_cast<float>(node.rotation[1]);
        float qz = static_cast<float>(node.rotation[2]);
        float qw = static_cast<float>(node.rotation[3]);
        float xx = qx*qx, yy = qy*qy, zz = qz*qz;
        float xy = qx*qy, xz = qx*qz, yz = qy*qz;
        float wx = qw*qx, wy = qw*qy, wz = qw*qz;
        R = mat4(1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy),   0,
                 2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx),   0,
                 2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy), 0,
                 0,            0,            0,            1);
    }

    if (node.scale.size() == 3)
        S = mat4::scaling(static_cast<float>(node.scale[0]),
                          static_cast<float>(node.scale[1]),
                          static_cast<float>(node.scale[2]));
    return T * R * S;
}

// ---- Stride-aware accessor helpers ----
struct AccessorInfo {
    const unsigned char* base;
    size_t stride;
    size_t count;
};

static AccessorInfo getAccessor(const tinygltf::Model& model, int accessorIdx,
                                size_t defaultElementSize) {
    const tinygltf::Accessor& acc = model.accessors[accessorIdx];
    const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];
    const tinygltf::Buffer& buf = model.buffers[bv.buffer];
    size_t stride = (bv.byteStride > 0) ? bv.byteStride : defaultElementSize;
    return { buf.data.data() + bv.byteOffset + acc.byteOffset, stride, acc.count };
}

static inline const float* stridedFloat(const AccessorInfo& ai, size_t index) {
    return reinterpret_cast<const float*>(ai.base + index * ai.stride);
}

// ---- Load textures (with sampler info) ----
static void loadTextures(const tinygltf::Model& model, std::vector<Image>& textures) {
    textures.resize(model.textures.size());
    for (size_t i = 0; i < model.textures.size(); ++i) {
        const tinygltf::Texture& tex = model.textures[i];
        if (tex.source < 0 || tex.source >= static_cast<int>(model.images.size()))
            continue;
        const tinygltf::Image& img = model.images[tex.source];
        if (img.image.empty()) continue;

        Image& out = textures[i];
        out.width   = img.width;
        out.height  = img.height;
        out.channel = 4;
        size_t npix = static_cast<size_t>(img.width) * img.height;
        out.data.resize(npix * 4);

        if (img.bits == 8 && img.component >= 3) {
            int srcComp = img.component;
            for (size_t p = 0; p < npix; ++p) {
                out.data[p*4+0] = img.image[p*srcComp+0] / 255.0f;
                out.data[p*4+1] = img.image[p*srcComp+1] / 255.0f;
                out.data[p*4+2] = img.image[p*srcComp+2] / 255.0f;
                out.data[p*4+3] = (srcComp >= 4)
                    ? img.image[p*srcComp+3] / 255.0f
                    : 1.0f;
            }
        }

        // Read sampler wrap modes
        if (tex.sampler >= 0 && tex.sampler < static_cast<int>(model.samplers.size())) {
            const tinygltf::Sampler& samp = model.samplers[tex.sampler];
            out.wrapS = samp.wrapS;
            out.wrapT = samp.wrapT;
        }
    }
}

// ---- Helper: read texture index from a glTF extension textureInfo object ----
static int readTexIndex(const tinygltf::Value& ext, const char* key, int offset) {
    if (!ext.Has(key)) return -1;
    auto& info = ext.Get(key);
    if (!info.Has("index")) return -1;
    return info.Get("index").GetNumberAsInt() + offset;
}

// ---- Helper: read KHR_texture_transform from extension Value ----
static void readTextureTransform(const tinygltf::Value& tt, Material& mat) {
    if (tt.Has("offset") && tt.Get("offset").IsArray() && tt.Get("offset").ArrayLen() >= 2) {
        mat.uvOffsetX = static_cast<float>(tt.Get("offset").Get(0).GetNumberAsDouble());
        mat.uvOffsetY = static_cast<float>(tt.Get("offset").Get(1).GetNumberAsDouble());
    }
    if (tt.Has("scale") && tt.Get("scale").IsArray() && tt.Get("scale").ArrayLen() >= 2) {
        mat.uvScaleX = static_cast<float>(tt.Get("scale").Get(0).GetNumberAsDouble());
        mat.uvScaleY = static_cast<float>(tt.Get("scale").Get(1).GetNumberAsDouble());
    }
    if (tt.Has("rotation"))
        mat.uvRotation = static_cast<float>(tt.Get("rotation").GetNumberAsDouble());
}

// ---- Load materials (optionally collect names) ----
static void loadMaterials(const tinygltf::Model& model,
                          std::vector<Material>& materials,
                          int textureOffset,
                          std::vector<std::string>* outNames = nullptr) {
    materials.resize(model.materials.size());
    if (outNames) {
        outNames->resize(model.materials.size());
    }
    for (size_t i = 0; i < model.materials.size(); ++i) {
        const tinygltf::Material& gm = model.materials[i];
        Material& mat = materials[i];
        if (outNames) (*outNames)[i] = gm.name;

        const auto& pbr = gm.pbrMetallicRoughness;
        if (pbr.baseColorFactor.size() >= 3) {
            mat.baseColor = vec3(static_cast<float>(pbr.baseColorFactor[0]),
                                 static_cast<float>(pbr.baseColorFactor[1]),
                                 static_cast<float>(pbr.baseColorFactor[2]));
        }
        if (pbr.baseColorFactor.size() >= 4) {
            mat.baseAlpha = static_cast<float>(pbr.baseColorFactor[3]);
        }
        mat.metallic  = static_cast<float>(pbr.metallicFactor);
        mat.roughness = static_cast<float>(pbr.roughnessFactor);

        if (pbr.baseColorTexture.index >= 0) {
            mat.baseColorTexId = pbr.baseColorTexture.index + textureOffset;
            // Read KHR_texture_transform from baseColorTexture extensions
            auto bcExtIt = pbr.baseColorTexture.extensions.find("KHR_texture_transform");
            if (bcExtIt != pbr.baseColorTexture.extensions.end())
                readTextureTransform(bcExtIt->second, mat);
        }
        if (pbr.metallicRoughnessTexture.index >= 0)
            mat.metallicRoughnessTexId = pbr.metallicRoughnessTexture.index + textureOffset;
        if (gm.normalTexture.index >= 0) {
            mat.normalTexId = gm.normalTexture.index + textureOffset;
            mat.normalScale = static_cast<float>(gm.normalTexture.scale);
        }
        if (gm.emissiveTexture.index >= 0)
            mat.emissiveTexId = gm.emissiveTexture.index + textureOffset;
        if (gm.occlusionTexture.index >= 0) {
            mat.occlusionTexId = gm.occlusionTexture.index + textureOffset;
            mat.occlusionStrength = static_cast<float>(gm.occlusionTexture.strength);
        }

        if (gm.emissiveFactor.size() >= 3) {
            mat.emissive = vec3(static_cast<float>(gm.emissiveFactor[0]),
                                static_cast<float>(gm.emissiveFactor[1]),
                                static_cast<float>(gm.emissiveFactor[2]));
            if (mat.emissive.x > 0 || mat.emissive.y > 0 || mat.emissive.z > 0)
                mat.emissiveStrength = 1.0f;
        }

        mat.doubleSided = gm.doubleSided ? 1 : 0;

        // KHR_materials_emissive_strength
        auto it = gm.extensions.find("KHR_materials_emissive_strength");
        if (it != gm.extensions.end() && it->second.Has("emissiveStrength")) {
            mat.emissiveStrength = static_cast<float>(
                it->second.Get("emissiveStrength").GetNumberAsDouble());
        }

        // KHR_materials_ior
        auto iorIt = gm.extensions.find("KHR_materials_ior");
        if (iorIt != gm.extensions.end() && iorIt->second.Has("ior")) {
            mat.ior = static_cast<float>(iorIt->second.Get("ior").GetNumberAsDouble());
        }

        // KHR_materials_transmission
        {
            auto transIt = gm.extensions.find("KHR_materials_transmission");
            if (transIt != gm.extensions.end()) {
                if (transIt->second.Has("transmissionFactor"))
                    mat.specTrans = static_cast<float>(
                        transIt->second.Get("transmissionFactor").GetNumberAsDouble());
                mat.transmissionTexId = readTexIndex(transIt->second, "transmissionTexture", textureOffset);
            }
        }

        // KHR_materials_clearcoat
        {
            auto ccIt = gm.extensions.find("KHR_materials_clearcoat");
            if (ccIt != gm.extensions.end()) {
                if (ccIt->second.Has("clearcoatFactor"))
                    mat.clearcoat = static_cast<float>(
                        ccIt->second.Get("clearcoatFactor").GetNumberAsDouble());
                if (ccIt->second.Has("clearcoatRoughnessFactor"))
                    mat.clearcoatRoughness = static_cast<float>(
                        ccIt->second.Get("clearcoatRoughnessFactor").GetNumberAsDouble());
                mat.clearcoatTexId          = readTexIndex(ccIt->second, "clearcoatTexture", textureOffset);
                mat.clearcoatRoughnessTexId = readTexIndex(ccIt->second, "clearcoatRoughnessTexture", textureOffset);
                mat.clearcoatNormalTexId    = readTexIndex(ccIt->second, "clearcoatNormalTexture", textureOffset);
            }
        }

        // KHR_materials_sheen
        {
            auto sheenIt = gm.extensions.find("KHR_materials_sheen");
            if (sheenIt != gm.extensions.end()) {
                if (sheenIt->second.Has("sheenRoughnessFactor"))
                    mat.sheen = static_cast<float>(
                        sheenIt->second.Get("sheenRoughnessFactor").GetNumberAsDouble());
                if (sheenIt->second.Has("sheenColorFactor")) {
                    auto& arr = sheenIt->second.Get("sheenColorFactor");
                    if (arr.IsArray() && arr.ArrayLen() >= 3) {
                        mat.sheenColorR = static_cast<float>(arr.Get(0).GetNumberAsDouble());
                        mat.sheenColorG = static_cast<float>(arr.Get(1).GetNumberAsDouble());
                        mat.sheenColorB = static_cast<float>(arr.Get(2).GetNumberAsDouble());
                    }
                }
                mat.sheenColorTexId     = readTexIndex(sheenIt->second, "sheenColorTexture", textureOffset);
                mat.sheenRoughnessTexId = readTexIndex(sheenIt->second, "sheenRoughnessTexture", textureOffset);
            }
        }

        // KHR_materials_specular
        {
            auto specIt = gm.extensions.find("KHR_materials_specular");
            if (specIt != gm.extensions.end()) {
                if (specIt->second.Has("specularFactor"))
                    mat.specularFactor = static_cast<float>(
                        specIt->second.Get("specularFactor").GetNumberAsDouble());
                if (specIt->second.Has("specularColorFactor")) {
                    auto& arr = specIt->second.Get("specularColorFactor");
                    if (arr.IsArray() && arr.ArrayLen() >= 3) {
                        mat.specularColorR = static_cast<float>(arr.Get(0).GetNumberAsDouble());
                        mat.specularColorG = static_cast<float>(arr.Get(1).GetNumberAsDouble());
                        mat.specularColorB = static_cast<float>(arr.Get(2).GetNumberAsDouble());
                    }
                }
            }
        }

        // KHR_materials_volume
        {
            auto volIt = gm.extensions.find("KHR_materials_volume");
            if (volIt != gm.extensions.end()) {
                if (volIt->second.Has("thicknessFactor"))
                    mat.thicknessFactor = static_cast<float>(
                        volIt->second.Get("thicknessFactor").GetNumberAsDouble());
                if (volIt->second.Has("attenuationDistance"))
                    mat.attenuationDistance = static_cast<float>(
                        volIt->second.Get("attenuationDistance").GetNumberAsDouble());
                if (volIt->second.Has("attenuationColor")) {
                    auto& arr = volIt->second.Get("attenuationColor");
                    if (arr.IsArray() && arr.ArrayLen() >= 3) {
                        mat.attColorR = static_cast<float>(arr.Get(0).GetNumberAsDouble());
                        mat.attColorG = static_cast<float>(arr.Get(1).GetNumberAsDouble());
                        mat.attColorB = static_cast<float>(arr.Get(2).GetNumberAsDouble());
                    }
                }
            }
        }

        // KHR_materials_anisotropy
        {
            auto aniIt = gm.extensions.find("KHR_materials_anisotropy");
            if (aniIt != gm.extensions.end()) {
                if (aniIt->second.Has("anisotropyStrength"))
                    mat.anisotropic = static_cast<float>(
                        aniIt->second.Get("anisotropyStrength").GetNumberAsDouble());
                if (aniIt->second.Has("anisotropyRotation"))
                    mat.anisotropyRotation = static_cast<float>(
                        aniIt->second.Get("anisotropyRotation").GetNumberAsDouble());
            }
        }

        // Alpha mode
        if (gm.alphaMode == "MASK") {
            mat.alphaMode   = 1;
            mat.alphaCutoff = static_cast<float>(gm.alphaCutoff);
        } else if (gm.alphaMode == "BLEND") {
            mat.alphaMode = 2;
        }
    }
}

// ---- Process a single mesh primitive ----
static void processPrimitive(const tinygltf::Model& model,
                             const tinygltf::Primitive& prim,
                             const mat4& worldTransform,
                             int materialOffset,
                             std::vector<Vertex>& vertices,
                             std::vector<TriangleFace>& faces) {
    if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1) return;

    // Positions (required)
    auto posIt = prim.attributes.find("POSITION");
    if (posIt == prim.attributes.end()) return;
    int posIdx = posIt->second;
    size_t vertCount = model.accessors[posIdx].count;
    AccessorInfo posAI = getAccessor(model, posIdx, 3 * sizeof(float));

    // Normals (optional, stride-aware)
    AccessorInfo normAI{};
    bool hasNorm = false;
    auto normIt = prim.attributes.find("NORMAL");
    if (normIt != prim.attributes.end()) {
        normAI  = getAccessor(model, normIt->second, 3 * sizeof(float));
        hasNorm = true;
    }

    // UVs (optional, stride-aware)
    AccessorInfo uvAI{};
    bool hasUV = false;
    auto uvIt = prim.attributes.find("TEXCOORD_0");
    if (uvIt != prim.attributes.end()) {
        uvAI  = getAccessor(model, uvIt->second, 2 * sizeof(float));
        hasUV = true;
    }

    // Tangents (optional, stride-aware)
    AccessorInfo tanAI{};
    bool hasTan = false;
    auto tanIt = prim.attributes.find("TANGENT");
    if (tanIt != prim.attributes.end()) {
        tanAI  = getAccessor(model, tanIt->second, 4 * sizeof(float));
        hasTan = true;
    }

    mat4 normalMatrix = worldTransform.inverse().transposed();

    uint32_t baseVertex = static_cast<uint32_t>(vertices.size());
    for (size_t v = 0; v < vertCount; ++v) {
        Vertex vert{};
        const float* p = stridedFloat(posAI, v);
        vert.position = worldTransform.transformPoint(vec3(p[0], p[1], p[2]));

        if (hasNorm) {
            const float* n = stridedFloat(normAI, v);
            vert.normal = normalize(normalMatrix.transformVector(vec3(n[0], n[1], n[2])));
        }

        if (hasUV) {
            const float* t = stridedFloat(uvAI, v);
            vert.uv = vec2(t[0], t[1]);
        }

        if (hasTan) {
            const float* tg = stridedFloat(tanAI, v);
            vec3 tanDir = normalize(worldTransform.transformVector(vec3(tg[0], tg[1], tg[2])));
            vert.tangent = vec4(tanDir.x, tanDir.y, tanDir.z, tg[3]);
        }

        vertices.push_back(vert);
    }

    int matId = (prim.material >= 0) ? prim.material + materialOffset : 0;

    // Indices
    if (prim.indices >= 0) {
        const tinygltf::Accessor& idxAcc = model.accessors[prim.indices];
        const tinygltf::BufferView& idxBv = model.bufferViews[idxAcc.bufferView];
        const tinygltf::Buffer& idxBuf = model.buffers[idxBv.buffer];
        const unsigned char* raw = idxBuf.data.data() + idxBv.byteOffset + idxAcc.byteOffset;

        size_t triCount = idxAcc.count / 3;
        for (size_t t = 0; t < triCount; ++t) {
            uint32_t i0, i1, i2;
            if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                const uint16_t* idx = reinterpret_cast<const uint16_t*>(raw);
                i0 = idx[t*3+0]; i1 = idx[t*3+1]; i2 = idx[t*3+2];
            } else if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                const uint32_t* idx = reinterpret_cast<const uint32_t*>(raw);
                i0 = idx[t*3+0]; i1 = idx[t*3+1]; i2 = idx[t*3+2];
            } else if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                i0 = raw[t*3+0]; i1 = raw[t*3+1]; i2 = raw[t*3+2];
            } else {
                continue;
            }
            faces.push_back({baseVertex + i0, baseVertex + i1, baseVertex + i2, matId});
        }
    } else {
        size_t triCount = vertCount / 3;
        for (size_t t = 0; t < triCount; ++t) {
            uint32_t i = static_cast<uint32_t>(t * 3);
            faces.push_back({baseVertex + i, baseVertex + i + 1, baseVertex + i + 2, matId});
        }
    }

    // Generate normals for faces that were missing them
    if (!hasNorm) {
        for (size_t v = 0; v < vertCount; ++v)
            vertices[baseVertex + v].normal = vec3(0.0f);

        size_t faceStart = faces.size() - (prim.indices >= 0
            ? model.accessors[prim.indices].count / 3
            : vertCount / 3);
        for (size_t fi = faceStart; fi < faces.size(); ++fi) {
            const TriangleFace& f = faces[fi];
            vec3 e1 = vertices[f.v1].position - vertices[f.v0].position;
            vec3 e2 = vertices[f.v2].position - vertices[f.v0].position;
            vec3 fn = cross(e1, e2);
            vertices[f.v0].normal += fn;
            vertices[f.v1].normal += fn;
            vertices[f.v2].normal += fn;
        }
        for (size_t v = 0; v < vertCount; ++v) {
            vec3& n = vertices[baseVertex + v].normal;
            float len = n.length();
            if (len > 0.0f) n = n / len;
        }
    }
}

// ---- Load KHR_lights_punctual definitions ----
struct GltfLightDef {
    LightType type = LightType::Point;
    vec3 color     = {1,1,1};
    float intensity = 1.0f;
    float range     = 1e30f;
    float innerCone = 0.0f;
    float outerCone = 0.7853982f;  // pi/4
};

static std::vector<GltfLightDef> loadLightDefs(const tinygltf::Model& model) {
    std::vector<GltfLightDef> defs;
    auto it = model.extensions.find("KHR_lights_punctual");
    if (it == model.extensions.end()) return defs;
    if (!it->second.Has("lights")) return defs;
    auto& lightsArr = it->second.Get("lights");
    if (!lightsArr.IsArray()) return defs;

    for (size_t i = 0; i < lightsArr.ArrayLen(); ++i) {
        auto& ld = lightsArr.Get(static_cast<int>(i));
        GltfLightDef def;
        if (ld.Has("type")) {
            std::string t = ld.Get("type").Get<std::string>();
            if (t == "directional") def.type = LightType::Directional;
            else if (t == "spot")   def.type = LightType::Spot;
            else                    def.type = LightType::Point;
        }
        if (ld.Has("color") && ld.Get("color").IsArray() && ld.Get("color").ArrayLen() >= 3) {
            auto& c = ld.Get("color");
            def.color = vec3(static_cast<float>(c.Get(0).GetNumberAsDouble()),
                             static_cast<float>(c.Get(1).GetNumberAsDouble()),
                             static_cast<float>(c.Get(2).GetNumberAsDouble()));
        }
        if (ld.Has("intensity"))
            def.intensity = static_cast<float>(ld.Get("intensity").GetNumberAsDouble());
        if (ld.Has("range"))
            def.range = static_cast<float>(ld.Get("range").GetNumberAsDouble());
        if (ld.Has("spot")) {
            auto& spot = ld.Get("spot");
            if (spot.Has("innerConeAngle"))
                def.innerCone = static_cast<float>(spot.Get("innerConeAngle").GetNumberAsDouble());
            if (spot.Has("outerConeAngle"))
                def.outerCone = static_cast<float>(spot.Get("outerConeAngle").GetNumberAsDouble());
        }
        defs.push_back(def);
    }
    return defs;
}

// ---- Traverse node hierarchy ----
static void traverseNodes(const tinygltf::Model& model,
                          int nodeIdx, const mat4& parentTransform,
                          int materialOffset,
                          const std::vector<GltfLightDef>& lightDefs,
                          std::vector<Vertex>& vertices,
                          std::vector<TriangleFace>& faces,
                          std::vector<PunctualLight>& punctualLights,
                          Camera& camera, bool& cameraSet) {
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(model.nodes.size())) return;
    const tinygltf::Node& node = model.nodes[nodeIdx];
    mat4 local = nodeTransform(node);
    mat4 world = parentTransform * local;

    // Process mesh
    if (node.mesh >= 0 && node.mesh < static_cast<int>(model.meshes.size())) {
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        for (const auto& prim : mesh.primitives) {
            processPrimitive(model, prim, world, materialOffset, vertices, faces);
        }
    }

    // Process camera
    if (!cameraSet && node.camera >= 0 && node.camera < static_cast<int>(model.cameras.size())) {
        const tinygltf::Camera& cam = model.cameras[node.camera];
        if (cam.type == "perspective") {
            camera.position = world.transformPoint(vec3(0, 0, 0));
            vec3 fwd = normalize(world.transformVector(vec3(0, 0, -1)));
            camera.lookAt = camera.position + fwd;
            camera.up = normalize(world.transformVector(vec3(0, 1, 0)));
            camera.fov = static_cast<float>(cam.perspective.yfov * 180.0 / 3.14159265358979323846);
            if (cam.perspective.aspectRatio > 0.0) {
                camera.width  = static_cast<int>(cam.perspective.aspectRatio * camera.height);
            }
            cameraSet = true;
        }
    }

    // Process KHR_lights_punctual
    auto extIt = node.extensions.find("KHR_lights_punctual");
    if (extIt != node.extensions.end() && extIt->second.Has("light")) {
        int lightIdx = extIt->second.Get("light").GetNumberAsInt();
        if (lightIdx >= 0 && lightIdx < static_cast<int>(lightDefs.size())) {
            const GltfLightDef& def = lightDefs[lightIdx];
            PunctualLight pl;
            pl.type      = def.type;
            pl.color     = def.color;
            pl.intensity = def.intensity;
            pl.range     = def.range;
            pl.position  = world.transformPoint(vec3(0, 0, 0));
            pl.direction = normalize(world.transformVector(vec3(0, 0, -1)));
            pl.innerCone = std::cos(def.innerCone);
            pl.outerCone = std::cos(def.outerCone);
            punctualLights.push_back(pl);
        }
    }

    for (int child : node.children)
        traverseNodes(model, child, world, materialOffset, lightDefs,
                      vertices, faces, punctualLights, camera, cameraSet);
}

bool GltfSceneLoader::load(const std::string& path, HostScene& scene,
                           const GltfLoadOptions& opts) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    loader.SetImageLoader(LoadImageData, nullptr);
    loader.SetImageWriter(WriteImageData, nullptr);

    std::string ext = std::filesystem::path(path).extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    bool ok;
    if (ext == ".glb") {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    } else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }

    if (!warn.empty()) std::cerr << "glTF warning: " << warn << std::endl;
    if (!err.empty())  std::cerr << "glTF error: " << err << std::endl;
    if (!ok) {
        std::cerr << "GltfSceneLoader: failed to load " << path << std::endl;
        return false;
    }

    int textureOffset  = static_cast<int>(scene.textures.size());
    int materialOffset = static_cast<int>(scene.materials.size());

    // Load textures
    std::vector<Image> gltfTextures;
    loadTextures(model, gltfTextures);
    for (auto& t : gltfTextures)
        scene.textures.push_back(std::move(t));

    // Load materials
    std::vector<Material> gltfMaterials;
    loadMaterials(model, gltfMaterials, textureOffset, opts.materialNames);
    for (auto& m : gltfMaterials)
        scene.materials.push_back(std::move(m));

    // If no materials, add a default
    if (scene.materials.empty()) {
        scene.materials.push_back(Material{});
    }

    // Load KHR_lights_punctual definitions
    std::vector<GltfLightDef> lightDefs = loadLightDefs(model);

    // Traverse scene nodes
    bool cameraSet = !opts.loadCamera;
    int defaultScene = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (defaultScene < static_cast<int>(model.scenes.size())) {
        const tinygltf::Scene& s = model.scenes[defaultScene];
        for (int nodeIdx : s.nodes) {
            traverseNodes(model, nodeIdx, opts.transform, materialOffset, lightDefs,
                          scene.vertices, scene.faces, scene.punctualLights,
                          scene.camera, cameraSet);
        }
    }

    if (!lightDefs.empty()) {
        std::cout << "GltfSceneLoader: loaded " << scene.punctualLights.size()
                  << " punctual lights from KHR_lights_punctual" << std::endl;
    }

    // Auto-fit camera from geometry bounds when no camera in the glTF
    if (opts.autoFitCamera && !cameraSet && !scene.vertices.empty()) {
        vec3 bmin( 1e30f,  1e30f,  1e30f);
        vec3 bmax(-1e30f, -1e30f, -1e30f);
        for (const auto& v : scene.vertices) {
            bmin.x = std::fmin(bmin.x, v.position.x);
            bmin.y = std::fmin(bmin.y, v.position.y);
            bmin.z = std::fmin(bmin.z, v.position.z);
            bmax.x = std::fmax(bmax.x, v.position.x);
            bmax.y = std::fmax(bmax.y, v.position.y);
            bmax.z = std::fmax(bmax.z, v.position.z);
        }
        vec3 center = (bmin + bmax) * 0.5f;
        vec3 diag   = bmax - bmin;
        float radius = diag.length() * 0.5f;
        if (radius < 1e-5f) radius = 1.0f;

        float fovRad = scene.camera.fov * pt::Pi / 180.0f;
        float dist   = radius / std::tan(fovRad * 0.5f) * 1.2f;

        scene.camera.position = center + vec3(0.0f, radius * 0.3f, dist);
        scene.camera.lookAt   = center;
        scene.camera.up       = vec3(0, 1, 0);

        std::cout << "GltfSceneLoader: auto-fit camera  center=("
                  << center.x << ", " << center.y << ", " << center.z
                  << ")  dist=" << dist << std::endl;
    }

    // Try loading default HDR environment if none is set
    if (opts.loadEnvironment && scene.hdrImage.width <= 0 && scene.hdrImage.height <= 0) {
        namespace fs = std::filesystem;
        fs::path scenePath(path);
        // Search upward from the scene file for a "models" directory containing the default HDR
        for (auto dir = fs::absolute(scenePath).parent_path(); dir.has_parent_path(); dir = dir.parent_path()) {
            fs::path candidate = dir / "models" / "kloofendal_48d_partly_cloudy_puresky_4k.hdr";
            if (fs::exists(candidate)) {
                ImageIO::load(scene.hdrImage, candidate.string());
                std::cout << "GltfSceneLoader: loaded default HDR environment: " << candidate.string() << std::endl;
                break;
            }
            if (dir == dir.parent_path()) break;
        }
    }

    // Add default punctual lights as fallback if still no lighting at all
    bool hasEnv      = (scene.hdrImage.width > 0 && scene.hdrImage.height > 0);
    bool hasEmissive = false;
    for (const auto& m : scene.materials) {
        if (isEmissive(m)) { hasEmissive = true; break; }
    }
    bool hasPunctual = !scene.punctualLights.empty();

    if (opts.addDefaultLights && !hasEnv && !hasEmissive && !hasPunctual) {
        PunctualLight key;
        key.type      = LightType::Directional;
        key.direction = normalize(vec3(-0.5f, -0.8f, -0.6f));
        key.color     = vec3(1.0f, 0.95f, 0.9f);
        key.intensity = 3.0f;
        scene.punctualLights.push_back(key);

        PunctualLight fill;
        fill.type      = LightType::Directional;
        fill.direction = normalize(vec3(0.6f, -0.3f, 0.5f));
        fill.color     = vec3(0.7f, 0.8f, 1.0f);
        fill.intensity = 1.0f;
        scene.punctualLights.push_back(fill);

        std::cout << "GltfSceneLoader: added default directional lights (no lighting found)" << std::endl;
    }

    std::cout << "GltfSceneLoader: " << path << " => "
              << scene.faces.size() << " triangles, "
              << scene.vertices.size() << " vertices, "
              << scene.materials.size() << " materials, "
              << scene.textures.size() << " textures" << std::endl;

    return true;
}

} // namespace pt
