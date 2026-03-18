#include "math/sampling.h"
#include "lights/light_sampler.h"

#include <cmath>
#include <iostream>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) do { \
    if (expr) { ++g_pass; } else { \
        std::cerr << "FAIL: " #expr " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        ++g_fail; } \
} while(0)

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

static void test_cosine_hemisphere_sample() {
    pt::vec3 dir = pt::sampleCosineHemisphere(0.25f, 0.5f);
    CHECK(dir.z >= 0.0f);
    CHECK(approx(dir.length(), 1.0f, 1e-4f));
}

static void test_power_heuristic() {
    CHECK(approx(pt::powerHeuristic(0.25f, 0.25f), 0.5f));
    CHECK(approx(pt::powerHeuristic(0.2f, 0.8f), 0.05882353f, 1e-6f));
}

static void addTriangle(std::vector<pt::Vertex>& verts,
                        std::vector<pt::TriangleFace>& faces,
                        pt::vec3 p0, pt::vec3 p1, pt::vec3 p2,
                        int matId) {
    pt::vec3 n = pt::normalize(pt::cross(p1 - p0, p2 - p0));
    uint32_t base = static_cast<uint32_t>(verts.size());
    verts.push_back({p0, n, {0,0}});
    verts.push_back({p1, n, {1,0}});
    verts.push_back({p2, n, {0,1}});
    faces.push_back({base, base+1, base+2, matId});
}

static void test_emissive_triangle_sampler() {
    std::vector<pt::Material> materials(3);
    materials[0].baseColor = {1.0f, 1.0f, 1.0f};
    materials[1].baseColor = {1.0f, 1.0f, 1.0f};
    materials[1].emissive  = {2.0f, 2.0f, 2.0f};
    materials[2].baseColor = {0.5f, 0.5f, 0.5f};
    materials[2].emissive  = {1.0f, 1.0f, 1.0f};

    std::vector<pt::Vertex> vertices;
    std::vector<pt::TriangleFace> faces;
    addTriangle(vertices, faces, pt::vec3(0,0,0), pt::vec3(1,0,0), pt::vec3(0,1,0), 0);
    addTriangle(vertices, faces, pt::vec3(0,0,0), pt::vec3(1,0,0), pt::vec3(0,1,0), 1);
    addTriangle(vertices, faces, pt::vec3(0,0,0), pt::vec3(2,0,0), pt::vec3(0,1,0), 2);

    pt::HostLightSampler sampler;
    pt::buildEmissiveTriangleSampler(vertices, faces, materials, sampler);

    CHECK(sampler.emissiveTriangles.size() == 2);
    CHECK(sampler.triangleSelectionPmf.size() == faces.size());
    CHECK(approx(sampler.triangleSelectionPmf[0], 0.0f));
    CHECK(approx(sampler.triangleSelectionPmf[1], 2.0f / 3.0f, 1e-5f));
    CHECK(approx(sampler.triangleSelectionPmf[2], 1.0f / 3.0f, 1e-5f));
    CHECK(approx(sampler.emissiveTriangles.back().cdf, 1.0f));
}

static void test_environment_sampler() {
    pt::Image hdr;
    hdr.width   = 2;
    hdr.height  = 1;
    hdr.channel = 4;
    hdr.data = {
        1.0f, 1.0f, 1.0f, 1.0f,
        3.0f, 3.0f, 3.0f, 1.0f
    };

    pt::HostLightSampler sampler;
    pt::buildEnvironmentSampler(hdr, sampler);

    CHECK(sampler.envWidth == 2);
    CHECK(sampler.envHeight == 1);
    CHECK(sampler.envPmf.size() == 2);
    CHECK(sampler.envAlias.size() == 2);
    CHECK(approx(sampler.envPmf[0], 0.25f, 1e-5f));
    CHECK(approx(sampler.envPmf[1], 0.75f, 1e-5f));
}

static void test_alias_table() {
    pt::Image hdr;
    hdr.width   = 4;
    hdr.height  = 1;
    hdr.channel = 4;
    hdr.data = {
        1.0f, 0.0f, 0.0f, 1.0f,
        2.0f, 0.0f, 0.0f, 1.0f,
        3.0f, 0.0f, 0.0f, 1.0f,
        4.0f, 0.0f, 0.0f, 1.0f
    };

    pt::HostLightSampler sampler;
    pt::buildEnvironmentSampler(hdr, sampler);

    CHECK(sampler.envAlias.size() == 4);
    for (auto& e : sampler.envAlias) {
        CHECK(e.prob >= 0.0f && e.prob <= 1.0f);
        CHECK(e.alias < 4);
    }
}

int main() {
    test_cosine_hemisphere_sample();
    test_power_heuristic();
    test_emissive_triangle_sampler();
    test_environment_sampler();
    test_alias_table();

    std::cout << "Sampling tests: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
