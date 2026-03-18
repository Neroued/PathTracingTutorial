#include "renderer.h"
#include "scene/scene_manager.h"
#include "integrators/integrator.h"
#include "io/image_io.h"
#include "cuda/check.h"

#include <iostream>
#include <iomanip>
#include <chrono>

namespace pt {

Renderer::Renderer(SceneManager& scene, std::unique_ptr<Integrator> integrator)
    : scene_(scene), integrator_(std::move(integrator))
{
    const Camera& cam = scene_.camera();
    film_.init(cam.width, cam.height);
}

uint32_t Renderer::renderBatch() {
    auto& dev   = scene_.deviceScene();
    auto view   = dev.view();
    auto params = dev.buildParams();

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    integrator_->launch(film_.newSurface(), film_.accSurface(), view, params);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float traceMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&traceMs, t0, t1));
    lastBatchMs_ = traceMs;
    totalTraceMs_ += traceMs;
    batchCount_++;

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    dev.frameCount++;
    return currentSpp();
}

uint32_t Renderer::currentSpp() const {
    auto& dev = scene_.deviceScene();
    return dev.frameCount * dev.samplePerFrame;
}

uint32_t Renderer::targetSpp() const {
    return scene_.deviceScene().targetSamples;
}

bool Renderer::isComplete() const {
    return currentSpp() >= static_cast<uint32_t>(targetSpp());
}

int Renderer::samplePerFrame() const {
    return scene_.deviceScene().samplePerFrame;
}

void Renderer::printTimingReport() const {
    if (batchCount_ == 0) return;
    std::cout << "\n=== GPU Timing Report (" << batchCount_ << " batches) ===\n"
              << "  Path trace + accum: " << totalTraceMs_ << " ms\n"
              << "  Avg per batch:      " << (totalTraceMs_ / batchCount_) << " ms\n";
}

void Renderer::resetAccumulation() {
    scene_.deviceScene().frameCount = 0;
    film_.clear();
    totalTraceMs_ = 0.0f;
    lastBatchMs_  = 0.0f;
    batchCount_   = 0;
}

void Renderer::resize(int w, int h) {
    if (w == film_.width() && h == film_.height()) return;
    scene_.deviceScene().camera.width  = w;
    scene_.deviceScene().camera.height = h;
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
