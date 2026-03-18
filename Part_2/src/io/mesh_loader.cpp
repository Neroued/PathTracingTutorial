#include "io/mesh_loader.h"
#include "tiny_obj_loader.h"

#include <iostream>
#include <unordered_map>

namespace pt {

struct ObjVertexKey {
    int vi, ni, ti;
    bool operator==(const ObjVertexKey& o) const { return vi == o.vi && ni == o.ni && ti == o.ti; }
};

struct ObjVertexHash {
    size_t operator()(const ObjVertexKey& k) const {
        size_t h = std::hash<int>()(k.vi);
        h ^= std::hash<int>()(k.ni) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.ti) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

bool MeshLoader::loadObj(const std::string& filename,
                         int materialID,
                         const mat4& transform,
                         std::vector<Vertex>& outVertices,
                         std::vector<TriangleFace>& outFaces)
{
    tinyobj::ObjReaderConfig cfg;
    cfg.mtl_search_path = "./";

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename, cfg)) {
        if (!reader.Error().empty())
            std::cerr << "MeshLoader: " << reader.Error() << std::endl;
        return false;
    }
    if (!reader.Warning().empty())
        std::cerr << "MeshLoader: " << reader.Warning() << std::endl;

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();
    const mat4 invT    = transform.inverse().transposed();

    uint32_t vertexBase = static_cast<uint32_t>(outVertices.size());
    std::unordered_map<ObjVertexKey, uint32_t, ObjVertexHash> vertexMap;

    auto getOrCreateVertex = [&](const tinyobj::index_t& idx) -> uint32_t {
        ObjVertexKey key{idx.vertex_index, idx.normal_index, idx.texcoord_index};
        auto it = vertexMap.find(key);
        if (it != vertexMap.end()) return it->second;

        Vertex vtx{};

        float vx = attrib.vertices[3 * idx.vertex_index + 0];
        float vy = attrib.vertices[3 * idx.vertex_index + 1];
        float vz = attrib.vertices[3 * idx.vertex_index + 2];
        vtx.position = transform.transformPoint(vec3(vx, vy, vz));

        if (idx.normal_index >= 0) {
            float nx = attrib.normals[3 * idx.normal_index + 0];
            float ny = attrib.normals[3 * idx.normal_index + 1];
            float nz = attrib.normals[3 * idx.normal_index + 2];
            vtx.normal = normalize(invT.transformVector(vec3(nx, ny, nz)));
        }

        if (idx.texcoord_index >= 0) {
            vtx.uv.x = attrib.texcoords[2 * idx.texcoord_index + 0];
            vtx.uv.y = attrib.texcoords[2 * idx.texcoord_index + 1];
        }

        uint32_t newIdx = static_cast<uint32_t>(outVertices.size());
        outVertices.push_back(vtx);
        vertexMap[key] = newIdx;
        return newIdx;
    };

    for (size_t s = 0; s < shapes.size(); ++s) {
        size_t indexOffset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
            size_t fv = shapes[s].mesh.num_face_vertices[f];
            if (fv != 3) { indexOffset += fv; continue; }

            uint32_t vi0 = getOrCreateVertex(shapes[s].mesh.indices[indexOffset + 0]);
            uint32_t vi1 = getOrCreateVertex(shapes[s].mesh.indices[indexOffset + 1]);
            uint32_t vi2 = getOrCreateVertex(shapes[s].mesh.indices[indexOffset + 2]);

            bool needNormal = (shapes[s].mesh.indices[indexOffset].normal_index < 0);
            if (needNormal) {
                vec3 faceN = normalize(cross(
                    outVertices[vi1].position - outVertices[vi0].position,
                    outVertices[vi2].position - outVertices[vi0].position));
                outVertices[vi0].normal = faceN;
                outVertices[vi1].normal = faceN;
                outVertices[vi2].normal = faceN;
            }

            TriangleFace face{};
            face.v0 = vi0;
            face.v1 = vi1;
            face.v2 = vi2;
            face.materialID = materialID;
            outFaces.push_back(face);

            indexOffset += fv;
        }
    }

    std::cout << "MeshLoader: loaded " << filename
              << " (" << (outVertices.size() - vertexBase) << " vertices)" << std::endl;
    return true;
}

} // namespace pt
