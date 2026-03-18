#pragma once

#include <cstdint>

struct GLFWwindow;

namespace pt {

struct RenderStats {
    uint32_t currentSpp   = 0;
    uint32_t targetSpp    = 0;
    float    elapsedSec   = 0.0f;
    float    lastBatchMs  = 0.0f;
    int      samplePerFrame = 1;
    int      filmWidth    = 0;
    int      filmHeight   = 0;
    uint32_t numVertices  = 0;
    uint32_t numFaces     = 0;
    uint32_t numBvhNodes  = 0;
    uint32_t numMaterials = 0;
    uint32_t numEmissiveTriangles = 0;
    bool     paused       = false;
    bool     complete     = false;
    bool     mouseControl = false;
    float    fov          = 90.0f;
};

struct UIActions {
    bool settingsChanged    = false;
    bool requestSave        = false;
    bool requestSaveScene   = false;
    bool togglePause        = false;
    bool requestReset       = false;
    bool toggleMouseControl = false;
    bool requestResize      = false;
    int  newWidth           = 0;
    int  newHeight          = 0;
};

class ImGuiLayer {
public:
    ImGuiLayer() = default;
    ~ImGuiLayer();

    void init(GLFWwindow* window);
    void destroy();

    UIActions render(float& exposure, float& gamma,
                     int& editWidth, int& editHeight,
                     float& moveSpeed, float& fov,
                     const RenderStats& stats);

    void newFrame();
    void endFrame();

private:
    bool initialized_ = false;
};

} // namespace pt
