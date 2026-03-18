#include "renderer.h"
#include "scene/scene_loader.h"
#include "accel/bvh_builder.h"
#include "io/image_io.h"
#include "lights/light_sampler.h"
#include "cuda/check.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>

namespace pt {

void Renderer::loadScene(const std::string& path) {
    scenePath_ = path;
    SceneLoader::load(path, hostScene_);
}

void Renderer::saveScene(const Camera& cam) {
    if (scenePath_.empty()) return;
    namespace fs = std::filesystem;
    std::string ext = fs::path(scenePath_).extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext != ".json") {
        std::cerr << "Renderer::saveScene: can only save to .json scenes" << std::endl;
        return;
    }
    SceneLoader::saveCamera(scenePath_, cam);
}

void Renderer::buildAccel() {
    std::cout << "Building BVH..." << std::endl;
    BVHBuilder builder;
    hostScene_.bvh = builder.build(hostScene_.vertices, hostScene_.faces, 0,
                                   static_cast<uint32_t>(hostScene_.faces.size()));
    buildEmissiveTriangleSampler(hostScene_.vertices, hostScene_.faces,
                                 hostScene_.materials, hostScene_.lightSampler);
    buildEnvironmentSampler(hostScene_.hdrImage, hostScene_.lightSampler);
    std::cout << "BVH: " << hostScene_.bvh.nodes.size() << " nodes, "
              << hostScene_.faces.size() << " faces, "
              << hostScene_.vertices.size() << " vertices, "
              << hostScene_.lightSampler.emissiveTriangles.size() << " emissive triangles"
              << std::endl;
}

void Renderer::uploadToDevice() {
    deviceScene_.uploadFrom(hostScene_);
    film_.init(hostScene_.camera.width, hostScene_.camera.height);
    std::cout << "Scene uploaded to GPU (sampler: "
              << (hostScene_.settings.samplerType == 1 ? "sobol" : "pcg") << ")" << std::endl;
}

uint32_t Renderer::renderBatch() {
    auto view   = deviceScene_.view();
    auto params = deviceScene_.buildParams();

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    launchPathTraceKernel(film_.newSurface(), film_.accSurface(), view, params);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float traceMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&traceMs, t0, t1));
    lastBatchMs_ = traceMs;
    totalTraceMs_ += traceMs;
    batchCount_++;

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    deviceScene_.frameCount++;
    return currentSpp();
}

void Renderer::printTimingReport() const {
    if (batchCount_ == 0) return;
    std::cout << "\n=== GPU Timing Report (" << batchCount_ << " batches) ===\n"
              << "  Path trace + accum: " << totalTraceMs_ << " ms\n"
              << "  Avg per batch:      " << (totalTraceMs_ / batchCount_) << " ms\n";
}

void Renderer::resetAccumulation() {
    deviceScene_.frameCount = 0;
    film_.clear();
    totalTraceMs_ = 0.0f;
    lastBatchMs_  = 0.0f;
    batchCount_   = 0;
}

void Renderer::resize(int w, int h) {
    if (w == film_.width() && h == film_.height()) return;
    deviceScene_.camera.width  = w;
    deviceScene_.camera.height = h;
    film_.init(w, h);
    resetAccumulation();
}

void Renderer::renderAll() {
    auto start = std::chrono::high_resolution_clock::now();
    auto last = start;

    while (!isComplete()) {
        uint32_t spp = renderBatch();

        auto now     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last);
        if (elapsed.count() >= 500 || isComplete()) {
            auto total = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
            float pct  = 100.0f * spp / targetSpp();
            std::cout << std::fixed << std::setprecision(1)
                      << "\rRendering: " << pct << "% (" << spp << "/" << targetSpp()
                      << " spp) elapsed " << total.count() / 1000.0f << "s   " << std::flush;
            last = now;
        }
    }

    cudaDeviceSynchronize();
    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    std::cout << "\nDone. Total time: " << total.count() / 1000.0f << "s" << std::endl;
    printTimingReport();
}

void Renderer::saveImage(const std::string& path) {
    int w = film_.width();
    int h = film_.height();
    Image img;
    img.width   = w;
    img.height  = h;
    img.channel = 4;
    img.data.resize(static_cast<size_t>(w) * h * 4);

    film_.downloadAccumulated(img.data.data());

    if (ImageIO::write(img, path)) {
        std::cout << "Image saved: " << path << std::endl;
    } else {
        std::cerr << "Failed to save: " << path << std::endl;
    }
}

void Renderer::downloadCurrentImage(std::vector<float>& buf) const {
    int w = film_.width();
    int h = film_.height();
    buf.resize(static_cast<size_t>(w) * h * 4);
    film_.downloadAccumulated(buf.data());
}

std::vector<QualityMetrics> Renderer::runConvergenceBenchmark(
    const float* refData,
    const std::vector<int>& checkpoints)
{
    resetAccumulation();
    std::vector<QualityMetrics> results;
    size_t nextCheckpoint = 0;

    auto start = std::chrono::high_resolution_clock::now();

    while (nextCheckpoint < checkpoints.size()) {
        uint32_t spp = renderBatch();

        if (static_cast<int>(spp) >= checkpoints[nextCheckpoint]) {
            auto now = std::chrono::high_resolution_clock::now();
            float ms = static_cast<float>(
                std::chrono::duration_cast<std::chrono::microseconds>(now - start).count()) / 1000.0f;

            std::vector<float> current;
            downloadCurrentImage(current);

            float rmse = computeRMSE(current.data(), refData,
                                     film_.width(), film_.height());
            QualityMetrics m;
            m.spp    = static_cast<int>(spp);
            m.rmse   = rmse;
            m.psnr   = rmseToDb(rmse);
            m.timeMs = ms;
            results.push_back(m);

            ++nextCheckpoint;
        }
    }

    return results;
}

} // namespace pt
