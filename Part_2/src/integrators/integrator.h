#pragma once

#include <cuda_runtime.h>

namespace pt {

struct DeviceSceneView;
struct SceneParams;

class Integrator {
public:
    virtual ~Integrator() = default;

    virtual void launch(cudaSurfaceObject_t surfNew,
                        cudaSurfaceObject_t surfAcc,
                        const DeviceSceneView& scene,
                        const SceneParams& params) = 0;

    virtual const char* name() const = 0;
};

} // namespace pt
