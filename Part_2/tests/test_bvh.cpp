#include "geometry/vertex.h"
#include "accel/bvh_builder.h"

#include <cassert>
#include <iostream>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) do { \
    if (expr) { ++g_pass; } else { \
        std::cerr << "FAIL: " #expr " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        ++g_fail; } \
} while(0)

static void addTriangle(std::vector<pt::Vertex>& verts,
                        std::vector<pt::TriangleFace>& faces,
                        pt::vec3 p0, pt::vec3 p1, pt::vec3 p2) {
    pt::vec3 n = pt::normalize(pt::cross(p1 - p0, p2 - p0));
    uint32_t base = static_cast<uint32_t>(verts.size());
    verts.push_back({p0, n, {0,0}});
    verts.push_back({p1, n, {1,0}});
    verts.push_back({p2, n, {0,1}});
    faces.push_back({base, base+1, base+2, 0});
}

static void test_single_triangle() {
    std::vector<pt::Vertex> verts;
    std::vector<pt::TriangleFace> faces;
    addTriangle(verts, faces, pt::vec3(0,0,0), pt::vec3(1,0,0), pt::vec3(0,1,0));

    pt::BVHBuilder builder;
    pt::BVH bvh = builder.build(verts, faces, 0, 1);

    CHECK(bvh.nodes.size() >= 1);
    CHECK(bvh.nodes[0].isLeaf());
}

static void test_two_triangles() {
    std::vector<pt::Vertex> verts;
    std::vector<pt::TriangleFace> faces;
    addTriangle(verts, faces, pt::vec3(-1,0,0), pt::vec3(0,0,0), pt::vec3(-0.5f,1,0));
    addTriangle(verts, faces, pt::vec3(1,0,0),  pt::vec3(2,0,0), pt::vec3(1.5f,1,0));

    pt::BVHBuilder builder;
    pt::BVH bvh = builder.build(verts, faces, 0, 2);

    CHECK(bvh.nodes.size() >= 1);
    CHECK(bvh.nodes[0].isLeaf());
    CHECK(bvh.nodes[0].primCount() == 2);
}

static void test_five_triangles() {
    std::vector<pt::Vertex> verts;
    std::vector<pt::TriangleFace> faces;
    for (int i = 0; i < 5; ++i) {
        float x = static_cast<float>(i) * 3.0f;
        addTriangle(verts, faces, pt::vec3(x,0,0), pt::vec3(x+1,0,0), pt::vec3(x+0.5f,1,0));
    }

    pt::BVHBuilder builder;
    pt::BVH bvh = builder.build(verts, faces, 0, 5);

    CHECK(bvh.nodes.size() >= 3);
    CHECK(!bvh.nodes[0].isLeaf());
}

static void test_many_triangles() {
    std::vector<pt::Vertex> verts;
    std::vector<pt::TriangleFace> faces;
    for (int i = 0; i < 100; ++i) {
        float x = static_cast<float>(i);
        addTriangle(verts, faces, pt::vec3(x,0,0), pt::vec3(x+1,0,0), pt::vec3(x+0.5f,1,0));
    }

    pt::BVHBuilder builder;
    pt::BVH bvh = builder.build(verts, faces, 0, 100);

    CHECK(!bvh.nodes.empty());
    CHECK(!bvh.nodes[0].isLeaf());
    CHECK(faces.size() == 100);
}

int main() {
    test_single_triangle();
    test_two_triangles();
    test_five_triangles();
    test_many_triangles();

    std::cout << "BVH tests: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
