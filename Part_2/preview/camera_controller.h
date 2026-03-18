#pragma once

#include "camera/camera.h"
#include "math/vecmath.h"
#include "pt.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>

namespace pt {

class CameraController {
public:
    void init(const Camera& cam) {
        camera_ = cam;
        vec3 fwd = normalize(cam.lookAt - cam.position);
        focusDistance_ = (cam.lookAt - cam.position).length();
        if (focusDistance_ < 1e-6f) focusDistance_ = 1.0f;

        pitch_ = std::asin(std::clamp(fwd.y, -1.0f, 1.0f));
        yaw_   = std::atan2(fwd.z, fwd.x);
    }

    // Returns true if the camera was modified this frame.
    // dt: seconds since last frame (for keyboard movement).
    bool update(GLFWwindow* window, float scrollDelta, float dt) {
        bool changed = false;

        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        float dx = static_cast<float>(mx - lastX_);
        float dy = static_cast<float>(my - lastY_);

        bool leftDown   = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS;
        bool rightDown  = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS;
        bool middleDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

        // --- Left drag: Orbit around lookAt ---
        if (leftDown && !rightDown && !middleDown) {
            if (leftDragging_) {
                yaw_   += dx * rotateSensitivity_;
                pitch_ -= dy * rotateSensitivity_;
                pitch_ = std::clamp(pitch_, -Pi * 0.499f, Pi * 0.499f);
                rebuildPositionFromLookAt();
                changed = true;
            }
            leftDragging_ = true;
        } else {
            leftDragging_ = false;
        }

        // --- Right drag: Free look (rotate view, position stays) ---
        if (rightDown && !middleDown) {
            if (rightDragging_) {
                yaw_   += dx * rotateSensitivity_;
                pitch_ -= dy * rotateSensitivity_;
                pitch_ = std::clamp(pitch_, -Pi * 0.499f, Pi * 0.499f);
                rebuildLookAtFromPosition();
                changed = true;
            }
            rightDragging_ = true;
        } else {
            rightDragging_ = false;
        }

        // --- Middle drag: Pan ---
        if (middleDown) {
            if (middleDragging_) {
                vec3 r = right();
                vec3 u = up();
                float scale = focusDistance_ * panSensitivity_;
                vec3 offset = r * (-dx * scale) + u * (dy * scale);
                camera_.position = camera_.position + offset;
                camera_.lookAt   = camera_.lookAt + offset;
                changed = true;
            }
            middleDragging_ = true;
        } else {
            middleDragging_ = false;
        }

        lastX_ = mx;
        lastY_ = my;

        // --- Scroll: Dolly along view direction ---
        if (scrollDelta != 0.0f) {
            vec3 fwd = forward();
            float dolly = scrollDelta * moveSpeed_ * scrollSpeed_;
            camera_.position = camera_.position + fwd * dolly;
            camera_.lookAt   = camera_.lookAt + fwd * dolly;
            changed = true;
        }

        // --- WASD / QE: movement (only while right-click held) ---
        if (rightDown) {
            float speed = moveSpeed_ * dt;
            if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
                speed *= 0.2f;

            vec3 fwd = forward();
            vec3 r   = right();
            vec3 move(0.0f);

            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move = move + fwd * speed;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move = move - fwd * speed;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move = move + r * speed;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move = move - r * speed;
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) move = move + vec3(0, 1, 0) * speed;
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) move = move - vec3(0, 1, 0) * speed;

            if (move.length() > 1e-8f) {
                camera_.position = camera_.position + move;
                camera_.lookAt   = camera_.lookAt + move;
                changed = true;
            }
        }

        return changed;
    }

    const Camera& camera() const { return camera_; }
    void setResolution(int w, int h) { camera_.width = w; camera_.height = h; }

    float fov() const          { return camera_.fov; }
    void  setFov(float f)      { camera_.fov = std::clamp(f, 1.0f, 170.0f); }

    float moveSpeed_ = 1.0f;
    float rotateSensitivity_ = 0.003f;
    float panSensitivity_    = 0.002f;
    float scrollSpeed_       = 0.3f;

private:
    vec3 forward() const {
        return {std::cos(pitch_) * std::cos(yaw_),
                std::sin(pitch_),
                std::cos(pitch_) * std::sin(yaw_)};
    }
    vec3 right() const {
        vec3 fwd = forward();
        return normalize(cross(fwd, vec3(0, 1, 0)));
    }
    vec3 up() const {
        return cross(right(), forward());
    }

    // Orbit: lookAt fixed, position recalculated
    void rebuildPositionFromLookAt() {
        camera_.position = camera_.lookAt - forward() * focusDistance_;
    }
    // Free-look: position fixed, lookAt recalculated
    void rebuildLookAtFromPosition() {
        camera_.lookAt = camera_.position + forward() * focusDistance_;
    }

    Camera camera_;
    float  yaw_           = 0.0f;
    float  pitch_         = 0.0f;
    float  focusDistance_  = 1.0f;

    bool   leftDragging_   = false;
    bool   rightDragging_  = false;
    bool   middleDragging_ = false;
    double lastX_ = 0.0;
    double lastY_ = 0.0;
};

} // namespace pt
