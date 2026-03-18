#pragma once

#include "scene/scene.h"
#include "scene/device_scene.h"
#include "film/film.h"
#include "film/quality_metrics.h"

#include <string>
#include <cstdint>
#include <vector>

namespace pt {

// Declared in integrators/path.cu
void launchPathTraceKernel(cudaSurfaceObject_t surfNew,
                           cudaSurfaceObject_t surfAcc,
                           const DeviceSceneView& scene,
                           const SceneParams& params);

class Renderer {
public:
    void loadScene(const std::string& path);
    void buildAccel();
    void uploadToDevice();

    uint32_t renderBatch();
    void renderAll();
    void saveImage(const std::string& path);
    void saveScene(const Camera& cam);
    void printTimingReport() const;
    void resetAccumulation();
    void resize(int w, int h);

    // Queries
    const Camera&      camera()     const { return hostScene_.camera; }
    const std::string& scenePath()  const { return scenePath_; }
    uint32_t           currentSpp() const { return deviceScene_.frameCount * deviceScene_.samplePerFrame; }
    uint32_t           targetSpp()  const { return deviceScene_.targetSamples; }
    bool               isComplete() const { return currentSpp() >= static_cast<uint32_t>(targetSpp()); }
    const std::string& outputPath() const { return hostScene_.settings.outputFile; }

    // Performance
    float lastBatchMs()    const { return lastBatchMs_; }
    int   samplePerFrame() const { return deviceScene_.samplePerFrame; }

    // Scene statistics
    uint32_t numVertices()          const { return static_cast<uint32_t>(hostScene_.vertices.size()); }
    uint32_t numFaces()             const { return static_cast<uint32_t>(hostScene_.faces.size()); }
    uint32_t numBvhNodes()          const { return static_cast<uint32_t>(hostScene_.bvh.nodes.size()); }
    uint32_t numMaterials()         const { return static_cast<uint32_t>(hostScene_.materials.size()); }
    uint32_t numEmissiveTriangles() const { return static_cast<uint32_t>(hostScene_.lightSampler.emissiveTriangles.size()); }

    // Live camera update (for interactive preview)
    void updateCamera(const Camera& cam) { deviceScene_.camera = cam; }

    // Settings override (call after loadScene, before uploadToDevice)
    void setSpp(int spp)                   { hostScene_.settings.targetSamples = spp; }
    void setMaxDepth(int depth)            { hostScene_.settings.maxDepth = depth; }
    void setOutput(const std::string& p)   { hostScene_.settings.outputFile = p; }
    void setSampler(int type)              { hostScene_.settings.samplerType = type; }

    // Convergence benchmark: render at multiple SPP checkpoints and measure RMSE/PSNR
    std::vector<QualityMetrics> runConvergenceBenchmark(const float* refData,
                                                        const std::vector<int>& checkpoints);
    void downloadCurrentImage(std::vector<float>& buf) const;

    // Minimal Film view for preview layer
    int         filmWidth()    const { return film_.width(); }
    int         filmHeight()   const { return film_.height(); }
    cudaArray_t filmAccArray() const { return film_.accArray(); }

private:
    HostScene   hostScene_;
    DeviceScene deviceScene_;
    Film        film_;
    std::string scenePath_;

    float totalTraceMs_ = 0.0f;
    float lastBatchMs_  = 0.0f;
    int   batchCount_   = 0;
};

} // namespace pt
