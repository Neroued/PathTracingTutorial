#pragma once

#include "gl_display.h"
#include "imgui_layer.h"
#include "camera_controller.h"
#include <string>

struct GLFWwindow;
using GLFWscrollfun = void(*)(GLFWwindow*, double, double);

namespace pt {

class SceneManager;
class Renderer;

class PreviewWindow {
public:
    PreviewWindow(SceneManager& sceneManager, Renderer& renderer,
                  int w, int h,
                  const std::string& vertShader,
                  const std::string& fragShader);
    ~PreviewWindow();

    PreviewWindow(const PreviewWindow&)            = delete;
    PreviewWindow& operator=(const PreviewWindow&) = delete;

    void run();

private:
    GLFWwindow*   window_       = nullptr;
    GLDisplay     display_;
    ImGuiLayer    imgui_;
    SceneManager* sceneManager_ = nullptr;
    Renderer*     renderer_     = nullptr;

    int width_;
    int height_;
    int editWidth_;
    int editHeight_;
    std::string vertShaderPath_;
    std::string fragShaderPath_;
    bool paused_ = false;

    CameraController cameraCtrl_;
    bool  mouseControlEnabled_ = false;
    float moveSpeed_           = 1.0f;
    float scrollAccum_         = 0.0f;
    GLFWscrollfun prevScrollCb_ = nullptr;

    void initGLFW();
    void mainLoop();
    void handleResize(int w, int h);

    static void scrollCallback(GLFWwindow* w, double xoff, double yoff);
};

} // namespace pt
