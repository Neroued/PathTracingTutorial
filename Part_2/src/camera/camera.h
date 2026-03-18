#pragma once

#include "pt.h"
#include "math/vecmath.h"
#include "math/math_utils.h"
#include "core/ray.h"

#include <cuda_runtime.h>

namespace pt {

struct Camera {
    vec3  position      = {0, 0, 4};
    vec3  lookAt        = {0, 0, 0};
    vec3  up            = {0, 1, 0};
    float fov           = 90.0f;
    float aperture      = 0.0f;
    float focalDistance  = 10.0f;
    int   width         = 1600;
    int   height        = 1600;
};

// POD camera basis for __constant__ memory and kernel use.
// Built from Camera on the host, consumed by emitRay on the device.
struct CameraFrame {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
    float  tanHalfFov;
    float  aspect;
    float  aperture;
    float  focalDistance;
};

inline CameraFrame buildCameraFrame(const Camera& cam) {
    vec3 fwd = normalize(cam.lookAt - cam.position);
    vec3 rgt = normalize(cross(fwd, cam.up));
    vec3 cup = cross(rgt, fwd);

    CameraFrame f{};
    f.position   = {cam.position.x, cam.position.y, cam.position.z};
    f.forward    = {fwd.x, fwd.y, fwd.z};
    f.right      = {rgt.x, rgt.y, rgt.z};
    f.up         = {cup.x, cup.y, cup.z};
    f.tanHalfFov    = pt::tan(cam.fov * Pi / 360.0f);
    f.aspect        = static_cast<float>(cam.width) / cam.height;
    f.aperture      = cam.aperture;
    f.focalDistance  = cam.focalDistance;
    return f;
}

} // namespace pt
