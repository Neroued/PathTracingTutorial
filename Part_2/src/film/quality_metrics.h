#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace pt {

struct QualityMetrics {
    float rmse = 0.0f;
    float psnr = 0.0f;
    int   spp  = 0;
    float timeMs = 0.0f;
};

inline float computeRMSE(const float* img, const float* ref, int width, int height, int channels = 3) {
    double sumSqErr = 0.0;
    int count = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 4;
            for (int c = 0; c < channels; ++c) {
                double diff = static_cast<double>(img[idx + c]) - static_cast<double>(ref[idx + c]);
                sumSqErr += diff * diff;
                ++count;
            }
        }
    }
    return (count > 0) ? static_cast<float>(std::sqrt(sumSqErr / count)) : 0.0f;
}

inline float rmseToDb(float rmse) {
    if (rmse <= 0.0f) return 100.0f;
    return -20.0f * std::log10(rmse);
}

inline void writeConvergenceCSV(const std::string& path,
                                const std::vector<QualityMetrics>& data,
                                const std::string& label) {
    bool fileExists = false;
    {
        std::ifstream test(path);
        fileExists = test.good();
    }

    std::ofstream f(path, std::ios::app);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << path << " for writing" << std::endl;
        return;
    }

    if (!fileExists) {
        f << "sampler,spp,rmse,psnr_db,time_ms\n";
    }

    for (const auto& m : data) {
        f << label << ","
          << m.spp << ","
          << std::fixed << std::setprecision(6) << m.rmse << ","
          << std::fixed << std::setprecision(2) << m.psnr << ","
          << std::fixed << std::setprecision(1) << m.timeMs << "\n";
    }

    std::cout << "Convergence data appended to " << path << " (" << label << ", "
              << data.size() << " checkpoints)" << std::endl;
}

inline void printConvergenceTable(const std::vector<QualityMetrics>& data,
                                  const std::string& label) {
    std::cout << "\n=== Convergence: " << label << " ===\n";
    std::cout << std::setw(8) << "SPP"
              << std::setw(12) << "RMSE"
              << std::setw(12) << "PSNR(dB)"
              << std::setw(12) << "Time(ms)" << "\n";
    for (const auto& m : data) {
        std::cout << std::setw(8) << m.spp
                  << std::setw(12) << std::fixed << std::setprecision(6) << m.rmse
                  << std::setw(12) << std::fixed << std::setprecision(2) << m.psnr
                  << std::setw(12) << std::fixed << std::setprecision(1) << m.timeMs << "\n";
    }
}

} // namespace pt
