#pragma once

#include "scene/device_scene.h"
#include "film/film.h"
#include "film/quality_metrics.h"

#include <string>
#include <cstdint>
#include <vector>
#include <memory>

namespace pt {

class SceneManager;
class Integrator;

class Renderer {
public:
    Renderer(SceneManager& scene, std::unique_ptr<Integrator> integrator);

    uint32_t renderBatch();
    void renderAll();
    void saveImage(const std::string& path);
    void printTimingReport() const;
    void resetAccumulation();
    void resize(int w, int h);

    // Queries
    uint32_t currentSpp() const;
    uint32_t targetSpp()  const;
    bool     isComplete() const;

    // Performance
    float lastBatchMs()    const { return lastBatchMs_; }
    int   samplePerFrame() const;

    // Convergence benchmark
    std::vector<QualityMetrics> runConvergenceBenchmark(const float* refData,
                                                        const std::vector<int>& checkpoints);
    void downloadCurrentImage(std::vector<float>& buf) const;

    // Film access (for preview)
    int         filmWidth()    const { return film_.width(); }
    int         filmHeight()   const { return film_.height(); }
    cudaArray_t filmAccArray() const { return film_.accArray(); }

private:
    SceneManager&              scene_;
    std::unique_ptr<Integrator> integrator_;
    Film                       film_;

    float totalTraceMs_ = 0.0f;
    float lastBatchMs_  = 0.0f;
    int   batchCount_   = 0;
};

} // namespace pt
