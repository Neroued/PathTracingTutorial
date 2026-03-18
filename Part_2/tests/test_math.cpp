#include "math/vecmath.h"
#include "math/math_utils.h"
#include "camera/camera.h"

#include <cmath>
#include <cassert>
#include <iostream>

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) do { \
    if (expr) { ++g_pass; } else { \
        std::cerr << "FAIL: " #expr " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        ++g_fail; } \
} while(0)

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

static void test_vec3_basic() {
    pt::vec3 a(1, 2, 3);
    pt::vec3 b(4, 5, 6);

    auto c = a + b;
    CHECK(approx(c.x, 5) && approx(c.y, 7) && approx(c.z, 9));

    auto d = a - b;
    CHECK(approx(d.x, -3) && approx(d.y, -3) && approx(d.z, -3));

    auto s = a * 2.0f;
    CHECK(approx(s.x, 2) && approx(s.y, 4) && approx(s.z, 6));

    CHECK(approx(dot(a, b), 32.0f));
}

static void test_vec3_cross() {
    pt::vec3 x(1, 0, 0);
    pt::vec3 y(0, 1, 0);
    auto z = cross(x, y);
    CHECK(approx(z.x, 0) && approx(z.y, 0) && approx(z.z, 1));
}

static void test_vec3_normalize() {
    pt::vec3 v(3, 4, 0);
    auto n = normalize(v);
    CHECK(approx(n.length(), 1.0f));
    CHECK(approx(n.x, 0.6f) && approx(n.y, 0.8f));
}

static void test_camera_frame_identity() {
    pt::Camera cam;
    cam.position = {0, 0, 4};
    cam.lookAt   = {0, 0, 0};
    cam.up       = {0, 1, 0};
    cam.fov      = 90.0f;
    cam.width    = 800;
    cam.height   = 800;

    pt::CameraFrame f = pt::buildCameraFrame(cam);

    CHECK(approx(f.forward.z, -1.0f));
    CHECK(approx(f.right.x, 1.0f));
    CHECK(approx(f.up.y, 1.0f));
    CHECK(approx(f.tanHalfFov, 1.0f, 1e-4f));
    CHECK(approx(f.aspect, 1.0f));
}

static void test_camera_frame_off_axis() {
    pt::Camera cam;
    cam.position = {0, 0, 0};
    cam.lookAt   = {1, 0, 0};
    cam.up       = {0, 1, 0};
    cam.fov      = 60.0f;
    cam.width    = 1920;
    cam.height   = 1080;

    pt::CameraFrame f = pt::buildCameraFrame(cam);

    CHECK(approx(f.forward.x, 1.0f));
    CHECK(approx(f.forward.y, 0.0f));
    CHECK(approx(f.forward.z, 0.0f));
    CHECK(approx(f.tanHalfFov, std::tan(30.0f * pt::Pi / 180.0f), 1e-4f));
    CHECK(approx(f.aspect, 1920.0f / 1080.0f, 1e-4f));
}

int main() {
    test_vec3_basic();
    test_vec3_cross();
    test_vec3_normalize();
    test_camera_frame_identity();
    test_camera_frame_off_axis();

    std::cout << "Math tests: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
