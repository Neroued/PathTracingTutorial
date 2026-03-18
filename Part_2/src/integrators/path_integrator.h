#pragma once

#include "integrators/integrator.h"

namespace pt {

class PathIntegrator : public Integrator {
public:
    void launch(cudaSurfaceObject_t surfNew,
                cudaSurfaceObject_t surfAcc,
                const DeviceSceneView& scene,
                const SceneParams& params) override;

    const char* name() const override { return "Path"; }
};

} // namespace pt
