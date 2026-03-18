#include "imgui_layer.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <cstdio>

namespace pt {

static void formatDuration(float seconds, char* buf, size_t bufSize) {
    if (seconds < 0.0f) { snprintf(buf, bufSize, "--:--"); return; }
    int total = static_cast<int>(seconds);
    int h = total / 3600;
    int m = (total % 3600) / 60;
    int s = total % 60;
    if (h > 0)
        snprintf(buf, bufSize, "%d:%02d:%02d", h, m, s);
    else
        snprintf(buf, bufSize, "%d:%02d", m, s);
}

void ImGuiLayer::init(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    float xscale = 1.0f, yscale = 1.0f;
    glfwGetWindowContentScale(window, &xscale, &yscale);
    float dpiScale = xscale;

    ImFontConfig cfg;
    cfg.SizePixels = 13.0f * dpiScale;
    io.Fonts->AddFontDefault(&cfg);

    ImGui::StyleColorsDark();
    ImGui::GetStyle().ScaleAllSizes(dpiScale);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    initialized_ = true;
}

void ImGuiLayer::destroy() {
    if (!initialized_) return;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    initialized_ = false;
}

ImGuiLayer::~ImGuiLayer() { destroy(); }

void ImGuiLayer::newFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::endFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

UIActions ImGuiLayer::render(float& exposure, float& gamma,
                             int& editWidth, int& editHeight,
                             float& moveSpeed, float& fov,
                             const RenderStats& stats)
{
    UIActions actions{};

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Render Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // --- Progress bar ---
    float fraction = stats.targetSpp > 0
        ? static_cast<float>(stats.currentSpp) / stats.targetSpp
        : 0.0f;
    if (fraction > 1.0f) fraction = 1.0f;

    char overlay[64];
    snprintf(overlay, sizeof(overlay), "%u / %u spp (%.1f%%)",
             stats.currentSpp, stats.targetSpp, fraction * 100.0f);
    ImGui::ProgressBar(fraction, ImVec2(-1, 0), overlay);

    // --- Elapsed + ETA ---
    char elapsedStr[32], etaStr[32];
    formatDuration(stats.elapsedSec, elapsedStr, sizeof(elapsedStr));
    if (stats.complete) {
        snprintf(etaStr, sizeof(etaStr), "Done");
    } else if (fraction > 0.001f) {
        float eta = stats.elapsedSec * (1.0f - fraction) / fraction;
        formatDuration(eta, etaStr, sizeof(etaStr));
    } else {
        snprintf(etaStr, sizeof(etaStr), "--:--");
    }
    ImGui::Text("Elapsed: %s  |  ETA: %s", elapsedStr, etaStr);

    // --- Performance ---
    if (stats.lastBatchMs > 0.0f) {
        double totalSamples = static_cast<double>(stats.samplePerFrame)
                            * stats.filmWidth * stats.filmHeight;
        double mspp = totalSamples / (stats.lastBatchMs * 1e3);
        ImGui::Text("%.1f ms/batch  |  %.2f Mspp/s",
                    stats.lastBatchMs, mspp);
    }

    ImGui::Separator();

    // --- Tone mapping ---
    if (ImGui::SliderFloat("Exposure", &exposure, -10.0f, 5.0f, "%.2f"))
        actions.settingsChanged = true;
    if (ImGui::SliderFloat("Gamma", &gamma, 0.5f, 4.0f, "%.2f"))
        actions.settingsChanged = true;

    ImGui::Separator();

    // --- Controls ---
    if (stats.paused) {
        if (ImGui::Button("Resume")) actions.togglePause = true;
    } else {
        if (ImGui::Button("Pause"))  actions.togglePause = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) actions.requestReset = true;
    ImGui::SameLine();
    if (ImGui::Button("Save Image")) actions.requestSave = true;
    ImGui::SameLine();
    if (ImGui::Button("Save Scene")) actions.requestSaveScene = true;

    // --- Mouse control ---
    bool mc = stats.mouseControl;
    if (ImGui::Checkbox("Mouse Control", &mc))
        actions.toggleMouseControl = true;

    if (stats.mouseControl) {
        ImGui::SliderFloat("Move Speed", &moveSpeed, 0.01f, 100.0f, "%.2f",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::SliderFloat("FOV", &fov, 1.0f, 170.0f, "%.1f"))
            actions.settingsChanged = true;
        ImGui::TextDisabled("LMB orbit | RMB look+WASD | MMB pan | Scroll dolly");
    }

    // --- Resolution ---
    ImGui::Separator();
    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("Width", &editWidth, 0, 0);
    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("Height", &editHeight, 0, 0);
    if (editWidth  < 64) editWidth  = 64;
    if (editHeight < 64) editHeight = 64;
    if (ImGui::Button("Apply Resolution")) {
        if (editWidth != stats.filmWidth || editHeight != stats.filmHeight) {
            actions.requestResize = true;
            actions.newWidth  = editWidth;
            actions.newHeight = editHeight;
        }
    }

    // --- Scene Info ---
    if (ImGui::CollapsingHeader("Scene Info")) {
        ImGui::Text("Resolution: %d x %d", stats.filmWidth, stats.filmHeight);
        ImGui::Text("Vertices:   %u", stats.numVertices);
        ImGui::Text("Faces:      %u", stats.numFaces);
        ImGui::Text("BVH Nodes:  %u", stats.numBvhNodes);
        ImGui::Text("Materials:  %u", stats.numMaterials);
        ImGui::Text("Emissive:   %u", stats.numEmissiveTriangles);
    }

    ImGui::End();
    return actions;
}

} // namespace pt
