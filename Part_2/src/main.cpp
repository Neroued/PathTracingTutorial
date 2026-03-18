#include "renderer.h"
#include "film/quality_metrics.h"

#ifdef PT_HAS_PREVIEW
#    include "preview_window.h"
#endif

#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <vector>

static std::string findProjectRoot(const char* argv0) {
    namespace fs = std::filesystem;
    // Try from executable location first, then CWD
    for (auto start : { fs::path(argv0).parent_path(), fs::current_path() }) {
        for (auto dir = fs::absolute(start); dir.has_parent_path(); dir = dir.parent_path()) {
            if (fs::exists(dir / "CMakeLists.txt") && fs::exists(dir / "scenes"))
                return dir.string();
            if (dir == dir.parent_path()) break;
        }
    }
    return ".";
}

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    std::string projectRoot = findProjectRoot(argv[0]);

    std::string scenePath;
    bool preview = false;
    bool benchmark = false;
    std::string benchmarkRef;
    int sppOverride = -1;
    int samplerOverride = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--preview") { preview = true; continue; }
        if (arg == "--spp" && i + 1 < argc) { sppOverride = std::atoi(argv[++i]); continue; }
        if (arg == "--sampler" && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "pcg")  samplerOverride = 0;
            else if (s == "sobol") samplerOverride = 1;
            continue;
        }
        if (arg == "--benchmark") {
            benchmark = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                benchmarkRef = argv[++i];
            continue;
        }
        if (arg.rfind("--", 0) == 0) continue;
        if (scenePath.empty()) scenePath = arg;
    }

    if (scenePath.empty()) {
        scenePath = (fs::path(projectRoot) / "scenes" / "cornell_box.json").string();
        std::cout << "Usage: pt [scene.json|.gltf|.glb] [--preview] [--spp N] [--sampler pcg|sobol] [--benchmark [ref.hdr]]" << std::endl;
        std::cout << "No scene specified, using default: " << scenePath << std::endl;
    } else if (!fs::exists(scenePath)) {
        auto inProject = fs::path(projectRoot) / scenePath;
        if (fs::exists(inProject)) scenePath = inProject.string();
    }

    pt::Renderer renderer;
    renderer.loadScene(scenePath);
    if (sppOverride > 0)
        renderer.setSpp(sppOverride);
    if (samplerOverride >= 0)
        renderer.setSampler(samplerOverride);
    renderer.buildAccel();
    renderer.uploadToDevice();

#ifdef PT_HAS_PREVIEW
    if (preview) {
        namespace fs = std::filesystem;
        fs::path exePath(argv[0]);
        fs::path baseDir = exePath.parent_path();

        std::string vertPath = (baseDir / "preview" / "fullscreen_quad.vert").string();
        std::string fragPath = (baseDir / "preview" / "tone_mapping.frag").string();

        // Fallback: look relative to working directory
        if (!fs::exists(vertPath)) vertPath = "preview/fullscreen_quad.vert";
        if (!fs::exists(fragPath)) fragPath = "preview/tone_mapping.frag";

        pt::PreviewWindow win(renderer,
                              renderer.camera().width,
                              renderer.camera().height,
                              vertPath, fragPath);
        win.run();
        return 0;
    }
#else
    if (preview) {
        std::cerr << "Preview not compiled. Build with PT_BUILD_PREVIEW=ON" << std::endl;
    }
#endif

    if (benchmark) {
        namespace fs = std::filesystem;
        std::vector<int> checkpoints = {16, 32, 64, 128, 256, 512, 1024};
        std::string csvPath = "convergence.csv";

        // Render high-SPP reference in memory (avoids file flip / precision issues)
        std::cout << "=== Rendering reference image (4096 spp, sobol) ===" << std::endl;
        renderer.setSampler(1);
        renderer.setSpp(4096);
        renderer.uploadToDevice();
        renderer.renderAll();
        std::vector<float> refData;
        renderer.downloadCurrentImage(refData);
        renderer.saveImage("benchmark_ref.hdr");

        if (fs::exists(csvPath)) fs::remove(csvPath);

        // Benchmark PCG
        {
            std::cout << "\n=== Benchmark: PCG ===" << std::endl;
            renderer.setSampler(0);
            renderer.setSpp(checkpoints.back());
            renderer.uploadToDevice();
            auto results = renderer.runConvergenceBenchmark(refData.data(), checkpoints);
            pt::printConvergenceTable(results, "pcg");
            pt::writeConvergenceCSV(csvPath, results, "pcg");
        }

        // Benchmark Sobol
        {
            std::cout << "\n=== Benchmark: Sobol ===" << std::endl;
            renderer.setSampler(1);
            renderer.setSpp(checkpoints.back());
            renderer.uploadToDevice();
            auto results = renderer.runConvergenceBenchmark(refData.data(), checkpoints);
            pt::printConvergenceTable(results, "sobol");
            pt::writeConvergenceCSV(csvPath, results, "sobol");
        }

        return 0;
    }

    renderer.renderAll();
    renderer.saveImage(renderer.outputPath());

    return 0;
}
