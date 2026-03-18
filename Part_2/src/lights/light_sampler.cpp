#include "lights/light_sampler.h"
#include <numeric>
#include <queue>

namespace pt {

static std::vector<AliasEntry> buildAliasTable(const std::vector<float>& weights) {
    const uint32_t n = static_cast<uint32_t>(weights.size());
    if (n == 0) return {};

    float sum = 0.0f;
    for (float w : weights) sum += w;
    if (sum <= 0.0f) return {};

    std::vector<float> prob(n);
    for (uint32_t i = 0; i < n; ++i)
        prob[i] = weights[i] / sum * static_cast<float>(n);

    std::vector<AliasEntry> table(n);
    std::queue<uint32_t> small, large;

    for (uint32_t i = 0; i < n; ++i) {
        if (prob[i] < 1.0f)
            small.push(i);
        else
            large.push(i);
    }

    while (!small.empty() && !large.empty()) {
        uint32_t s = small.front(); small.pop();
        uint32_t l = large.front(); large.pop();

        table[s].prob  = prob[s];
        table[s].alias = l;

        prob[l] = (prob[l] + prob[s]) - 1.0f;
        if (prob[l] < 1.0f)
            small.push(l);
        else
            large.push(l);
    }

    while (!large.empty()) {
        uint32_t l = large.front(); large.pop();
        table[l].prob  = 1.0f;
        table[l].alias = l;
    }
    while (!small.empty()) {
        uint32_t s = small.front(); small.pop();
        table[s].prob  = 1.0f;
        table[s].alias = s;
    }

    return table;
}

void buildEmissiveTriangleSampler(const std::vector<Vertex>& vertices,
                                  const std::vector<TriangleFace>& faces,
                                  const std::vector<Material>& materials,
                                  HostLightSampler& sampler) {
    sampler.emissiveTriangles.clear();
    sampler.triangleSelectionPmf.assign(faces.size(), 0.0f);
    sampler.triangleAlias.clear();

    std::vector<float> weights;
    weights.reserve(faces.size());

    for (uint32_t faceIdx = 0; faceIdx < static_cast<uint32_t>(faces.size()); ++faceIdx) {
        const TriangleFace& face = faces[faceIdx];
        if (face.materialID < 0 || face.materialID >= static_cast<int>(materials.size()))
            continue;

        const Material& mat = materials[face.materialID];
        if (!isEmissive(mat)) continue;

        float area = faceArea(vertices[face.v0], vertices[face.v1], vertices[face.v2]);
        if (area <= 0.0f) continue;

        float weight = area * pt::max(luminance(emittedRadiance(mat)), 0.0f);
        if (weight <= 0.0f) continue;

        EmissiveTriangleRef ref{};
        ref.triangleIndex = faceIdx;
        sampler.emissiveTriangles.push_back(ref);
        weights.push_back(weight);
    }

    if (sampler.emissiveTriangles.empty()) return;

    float weightSum = 0.0f;
    for (float weight : weights) weightSum += weight;
    if (weightSum <= 0.0f) {
        sampler.emissiveTriangles.clear();
        return;
    }

    float accum = 0.0f;
    for (size_t i = 0; i < sampler.emissiveTriangles.size(); ++i) {
        float pmf = weights[i] / weightSum;
        accum += pmf;

        EmissiveTriangleRef& ref = sampler.emissiveTriangles[i];
        ref.selectionPmf = pmf;
        ref.cdf          = accum;
        sampler.triangleSelectionPmf[ref.triangleIndex] = pmf;
    }
    sampler.emissiveTriangles.back().cdf = 1.0f;

    sampler.triangleAlias = buildAliasTable(weights);
}

void buildEnvironmentSampler(const Image& hdrImage, HostLightSampler& sampler) {
    sampler.envPmf.clear();
    sampler.envAlias.clear();
    sampler.envWidth  = 0;
    sampler.envHeight = 0;

    if (hdrImage.width <= 0 || hdrImage.height <= 0 || hdrImage.channel < 3) return;

    const int texelCount = hdrImage.width * hdrImage.height;
    std::vector<float> weights(static_cast<size_t>(texelCount), 0.0f);

    for (int y = 0; y < hdrImage.height; ++y) {
        float theta    = Pi * (static_cast<float>(y) + 0.5f) / static_cast<float>(hdrImage.height);
        float sinTheta = pt::max(pt::sin(theta), 0.0f);

        for (int x = 0; x < hdrImage.width; ++x) {
            int idx = y * hdrImage.width + x;
            int px  = idx * hdrImage.channel;
            vec3 radiance(hdrImage.data[px + 0],
                          hdrImage.data[px + 1],
                          hdrImage.data[px + 2]);

            weights[idx] = pt::max(luminance(radiance), 0.0f) * sinTheta;
        }
    }

    float weightSum = 0.0f;
    for (float w : weights) weightSum += w;
    if (weightSum <= 0.0f) return;

    sampler.envWidth  = hdrImage.width;
    sampler.envHeight = hdrImage.height;
    sampler.envPmf.resize(static_cast<size_t>(texelCount));
    for (int idx = 0; idx < texelCount; ++idx)
        sampler.envPmf[idx] = weights[idx] / weightSum;

    sampler.envAlias = buildAliasTable(weights);
}

} // namespace pt
