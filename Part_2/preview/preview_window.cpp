#include "preview_window.h"
#include "renderer.h"

#include <imgui.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <chrono>
#include <algorithm>

namespace pt {

PreviewWindow::PreviewWindow(Renderer& renderer, int w, int h,
                             const std::string& vertShader,
                             const std::string& fragShader)
    : renderer_(&renderer), width_(w), height_(h),
      editWidth_(w), editHeight_(h),
      vertShaderPath_(vertShader), fragShaderPath_(fragShader)
{
    initGLFW();
    display_.init(w, h, vertShader, fragShader);
    imgui_.init(window_);

    glfwSetWindowUserPointer(window_, this);
    prevScrollCb_ = glfwSetScrollCallback(window_, scrollCallback);

    cameraCtrl_.init(renderer_->camera());
}

PreviewWindow::~PreviewWindow() {
    imgui_.destroy();
    display_.destroy();
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

void PreviewWindow::scrollCallback(GLFWwindow* w, double xoff, double yoff) {
    auto* self = static_cast<PreviewWindow*>(glfwGetWindowUserPointer(w));
    self->scrollAccum_ += static_cast<float>(yoff);
    if (self->prevScrollCb_)
        self->prevScrollCb_(w, xoff, yoff);
}

void PreviewWindow::initGLFW() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(width_, height_, "Path Tracer Preview", nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);
    glfwSetWindowSizeLimits(window_, 64, 64, GLFW_DONT_CARE, GLFW_DONT_CARE);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void PreviewWindow::handleResize(int w, int h) {
    if (w == width_ && h == height_) return;
    width_  = w;
    height_ = h;
    editWidth_  = w;
    editHeight_ = h;
    renderer_->resize(w, h);
    display_.resize(w, h);
    cameraCtrl_.setResolution(w, h);
}

void PreviewWindow::run() {
    mainLoop();
}

void PreviewWindow::mainLoop() {
    using Clock = std::chrono::high_resolution_clock;
    auto lastFrame = Clock::now();
    float activeTime = 0.0f;

    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();

        // Detect window resize from user dragging
        int winW, winH;
        glfwGetWindowSize(window_, &winW, &winH);
        if (winW > 0 && winH > 0 && (winW != width_ || winH != height_)) {
            handleResize(winW, winH);
            activeTime = 0.0f;
        }

        if (winW <= 0 || winH <= 0) {
            glfwSwapBuffers(window_);
            continue;
        }

        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - lastFrame).count();
        lastFrame = now;
        if (!paused_) activeTime += dt;

        // Start ImGui frame early so WantCaptureMouse is available
        imgui_.newFrame();

        // Camera controller
        cameraCtrl_.moveSpeed_ = moveSpeed_;
        if (mouseControlEnabled_ && !ImGui::GetIO().WantCaptureMouse) {
            if (cameraCtrl_.update(window_, scrollAccum_, dt)) {
                renderer_->updateCamera(cameraCtrl_.camera());
                renderer_->resetAccumulation();
                activeTime = 0.0f;
            }
        }
        scrollAccum_ = 0.0f;

        if (!paused_ && !renderer_->isComplete()) {
            renderer_->renderBatch();
        }

        display_.updateFromCudaArray(renderer_->filmAccArray(),
                                     renderer_->filmWidth(),
                                     renderer_->filmHeight());

        // Letterbox / pillarbox viewport
        int fbW, fbH;
        glfwGetFramebufferSize(window_, &fbW, &fbH);

        float imageAspect  = static_cast<float>(renderer_->filmWidth())
                           / std::max(renderer_->filmHeight(), 1);
        float windowAspect = static_cast<float>(fbW)
                           / std::max(fbH, 1);

        int vpW, vpH, vpX, vpY;
        if (windowAspect > imageAspect) {
            vpH = fbH;
            vpW = static_cast<int>(fbH * imageAspect);
            vpX = (fbW - vpW) / 2;
            vpY = 0;
        } else {
            vpW = fbW;
            vpH = static_cast<int>(fbW / imageAspect);
            vpX = 0;
            vpY = (fbH - vpH) / 2;
        }

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(vpX, vpY, vpW, vpH);

        display_.render();

        glViewport(0, 0, fbW, fbH);

        // ImGui widgets
        float exposure = display_.exposure();
        float gamma    = display_.gamma();

        RenderStats stats{};
        stats.currentSpp           = renderer_->currentSpp();
        stats.targetSpp            = renderer_->targetSpp();
        stats.elapsedSec           = activeTime;
        stats.lastBatchMs          = renderer_->lastBatchMs();
        stats.samplePerFrame       = renderer_->samplePerFrame();
        stats.filmWidth            = renderer_->filmWidth();
        stats.filmHeight           = renderer_->filmHeight();
        stats.numVertices          = renderer_->numVertices();
        stats.numFaces             = renderer_->numFaces();
        stats.numBvhNodes          = renderer_->numBvhNodes();
        stats.numMaterials         = renderer_->numMaterials();
        stats.numEmissiveTriangles = renderer_->numEmissiveTriangles();
        stats.paused               = paused_;
        stats.complete             = renderer_->isComplete();
        stats.mouseControl         = mouseControlEnabled_;
        stats.fov                  = cameraCtrl_.camera().fov;

        float fov = cameraCtrl_.fov();
        UIActions actions = imgui_.render(exposure, gamma,
                                          editWidth_, editHeight_,
                                          moveSpeed_, fov, stats);

        display_.setExposure(exposure);
        display_.setGamma(gamma);

        if (fov != cameraCtrl_.fov()) {
            cameraCtrl_.setFov(fov);
            renderer_->updateCamera(cameraCtrl_.camera());
            renderer_->resetAccumulation();
            activeTime = 0.0f;
        }

        if (actions.togglePause) paused_ = !paused_;
        if (actions.requestReset) {
            renderer_->resetAccumulation();
            activeTime = 0.0f;
            paused_ = false;
        }
        if (actions.requestSave) {
            renderer_->saveImage(renderer_->outputPath());
        }
        if (actions.requestSaveScene) {
            renderer_->saveScene(cameraCtrl_.camera());
        }
        if (actions.toggleMouseControl) {
            mouseControlEnabled_ = !mouseControlEnabled_;
        }
        if (actions.requestResize) {
            glfwSetWindowSize(window_, actions.newWidth, actions.newHeight);
            handleResize(actions.newWidth, actions.newHeight);
            activeTime = 0.0f;
        }

        imgui_.endFrame();

        glfwSwapBuffers(window_);
    }
}

} // namespace pt
