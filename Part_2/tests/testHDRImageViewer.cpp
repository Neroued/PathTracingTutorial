#include <QApplication>
#include <vector> // 使用 std::vector 来存储图像数据
#include <QTimer>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>

// 包含你的 ImageWidget 类定义
#include "HDRImageViewer.h"

// 如果 ImageWidget 的实现不在头文件中，确保链接其 .cpp 文件或库

// 函数：生成一个简单的测试图像数据 (浮点 RGB)
std::vector<float> generateTestData(int width, int height) {
    std::vector<float> data(width * height * 3); // 3 个通道 (R, G, B)

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // 计算当前像素在数组中的起始索引
            int index = (y * width + x) * 3;

            // 生成一个简单的渐变图案
            float r = static_cast<float>(x) / (width - 1);  // 红色从左到右渐变
            float g = static_cast<float>(y) / (height - 1); // 绿色从上到下渐变
            float b = 0.2f;                                 // 蓝色设为一个常数

            // 填充数据
            data[index + 0] = r; // R 通道
            data[index + 1] = g; // G 通道
            data[index + 2] = b; // B 通道
        }
    }
    return data;
}

// 函数：生成一个随时间变化的测试图像数据 (浮点 RGB)
std::vector<float> generateAnimatedData(int width, int height, float t) {
    std::vector<float> data(width * height * 3);

    const float wx = 0.12f;
    const float wy = 0.10f;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = (y * width + x) * 3;

            float r = static_cast<float>(x) / (width - 1);
            float g = static_cast<float>(y) / (height - 1);
            float b = 0.5f + 0.5f * std::sin(wx * x + wy * y + 0.5 * t);

            data[index + 0] = r;
            data[index + 1] = g;
            data[index + 2] = b;
        }
    }
    return data;
}

int main(int argc, char* argv[]) {
    // 确保请求的 OpenGL 版本和 Profile 与 ImageWidget 内部使用的匹配
    // ImageWidget 使用 QOpenGLFunctions_4_5_Core，所以请求 4.5 Core Profile
    QSurfaceFormat format;
    format.setVersion(4, 5);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format); // 应用到所有后续创建的 OpenGL Widget

    QApplication app(argc, argv);

    // 2. 定义图像尺寸并生成测试数据
    const int imageWidth             = 256;
    const int imageHeight            = 256;
    std::vector<float> testImageData = generateTestData(imageWidth, imageHeight);

    // 3. 创建 ImageWidget 实例
    HDRImageViewer hdrImageViewer;

    // 设置窗口标题和初始大小 (可选)
    hdrImageViewer.setWindowTitle("ImageWidget Test Display");

    hdrImageViewer.setImageData(testImageData.data(), imageWidth, imageHeight);

    // 后台线程：周期生成新图像并在 UI 线程更新
    std::atomic<bool> running{true};

    QObject::connect(&app, &QCoreApplication::aboutToQuit, [&]() { running = false; });
    QObject::connect(&hdrImageViewer, &QObject::destroyed, [&]() { running = false; });

    std::thread worker([&]() {
        int frame   = 0;
        const int w = imageWidth;
        const int h = imageHeight;
        while (running) {
            float t     = static_cast<float>(frame) * 0.15f;
            auto data   = generateAnimatedData(w + frame, h + frame, t);
            auto buffer = std::make_shared<std::vector<float>>(std::move(data));

            QMetaObject::invokeMethod(
                &hdrImageViewer,
                [buffer, w, h, &hdrImageViewer, frame]() {
                    hdrImageViewer.setImageData(buffer->data(), w + frame, h + frame);
                },
                Qt::QueuedConnection);

            ++frame;
            for (int i = 0; i < 5 && running; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    });

    hdrImageViewer.show();
    // 6. 启动 Qt 事件循环
    // 程序将在此处运行，处理事件（包括绘制事件），直到窗口关闭
    int rc = app.exec();

    // 退出时安全停止并回收线程
    running = false;
    if (worker.joinable()) { worker.join(); }

    return rc;
}