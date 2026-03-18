#pragma once

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>

namespace pt {

// Manages OpenGL display: PBO for CUDA-GL interop, fullscreen quad, tone mapping shader.
class GLDisplay {
public:
    GLDisplay() = default;
    ~GLDisplay();

    GLDisplay(const GLDisplay&)            = delete;
    GLDisplay& operator=(const GLDisplay&) = delete;

    void init(int w, int h, const std::string& vertPath, const std::string& fragPath);
    void resize(int w, int h);
    void destroy();

    void setExposure(float e) { exposure_ = e; }
    void setGamma(float g)    { gamma_ = g; }
    float exposure() const { return exposure_; }
    float gamma()    const { return gamma_; }

    // Copy from a CUDA array (float4) into the PBO, then update the GL texture.
    void updateFromCudaArray(cudaArray_t srcArray, int w, int h);

    // Draw the fullscreen quad with tone mapping.
    void render();

    // Accessors for low-level interop (used by PreviewWindow)
    cudaGraphicsResource_t& cudaPboRes() { return cudaPboRes_; }
    GLuint pbo()     const { return pbo_; }
    GLuint texture() const { return texture_; }

private:
    GLuint shaderProgram_ = 0;
    GLuint vao_           = 0;
    GLuint vbo_           = 0;
    GLuint texture_       = 0;
    GLuint pbo_           = 0;

    cudaGraphicsResource_t cudaPboRes_ = nullptr;

    int width_  = 0;
    int height_ = 0;

    float exposure_ = -2.8f;
    float gamma_    = 2.2f;

    GLuint compileShader(GLenum type, const std::string& path);
    GLuint linkProgram(GLuint vert, GLuint frag);
    void   initQuad();
};

} // namespace pt
