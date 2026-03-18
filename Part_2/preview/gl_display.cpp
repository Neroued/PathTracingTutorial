#include "gl_display.h"
#include "cuda/check.h"

#include <fstream>
#include <sstream>
#include <iostream>

namespace pt {

GLDisplay::~GLDisplay() { destroy(); }

void GLDisplay::destroy() {
    if (cudaPboRes_) {
        cudaGraphicsUnregisterResource(cudaPboRes_);
        cudaPboRes_ = nullptr;
    }
    if (pbo_)           { glDeleteBuffers(1, &pbo_);           pbo_ = 0; }
    if (texture_)       { glDeleteTextures(1, &texture_);      texture_ = 0; }
    if (vao_)           { glDeleteVertexArrays(1, &vao_);      vao_ = 0; }
    if (vbo_)           { glDeleteBuffers(1, &vbo_);           vbo_ = 0; }
    if (shaderProgram_) { glDeleteProgram(shaderProgram_);     shaderProgram_ = 0; }
}

void GLDisplay::init(int w, int h, const std::string& vertPath, const std::string& fragPath) {
    width_  = w;
    height_ = h;

    GLuint vs = compileShader(GL_VERTEX_SHADER,   vertPath);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragPath);
    shaderProgram_ = linkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);

    initQuad();

    // Create texture
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);

    // Create PBO for CUDA-GL interop
    size_t bufSize = static_cast<size_t>(w) * h * 4 * sizeof(float);
    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, bufSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboRes_, pbo_, cudaGraphicsMapFlagsWriteDiscard));
}

void GLDisplay::resize(int w, int h) {
    if (w == width_ && h == height_) return;

    if (cudaPboRes_) {
        cudaGraphicsUnregisterResource(cudaPboRes_);
        cudaPboRes_ = nullptr;
    }
    if (pbo_)     { glDeleteBuffers(1, &pbo_);      pbo_ = 0; }
    if (texture_) { glDeleteTextures(1, &texture_);  texture_ = 0; }

    width_  = w;
    height_ = h;

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);

    size_t bufSize = static_cast<size_t>(w) * h * 4 * sizeof(float);
    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, bufSize, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPboRes_, pbo_, cudaGraphicsMapFlagsWriteDiscard));
}

void GLDisplay::updateFromCudaArray(cudaArray_t srcArray, int w, int h) {
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPboRes_, 0));

    void*  mappedPtr  = nullptr;
    size_t mappedSize = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&mappedPtr, &mappedSize, cudaPboRes_));

    CUDA_CHECK(cudaMemcpy2DFromArray(
        mappedPtr, w * sizeof(float4),
        srcArray, 0, 0,
        w * sizeof(float4), h,
        cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPboRes_, 0));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void GLDisplay::render() {
    glUseProgram(shaderProgram_);
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glUniform1i(glGetUniformLocation(shaderProgram_, "imageTexture"), 0);
    glUniform1f(glGetUniformLocation(shaderProgram_, "gamma"),    gamma_);
    glUniform1f(glGetUniformLocation(shaderProgram_, "exposure"), exposure_);

    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glUseProgram(0);
}

void GLDisplay::initQuad() {
    float verts[] = {
        -1.f,  1.f, 0.f, 1.f,
        -1.f, -1.f, 0.f, 0.f,
         1.f, -1.f, 1.f, 0.f,
        -1.f,  1.f, 0.f, 1.f,
         1.f, -1.f, 1.f, 0.f,
         1.f,  1.f, 1.f, 1.f,
    };

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindVertexArray(0);
}

GLuint GLDisplay::compileShader(GLenum type, const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "GLDisplay: cannot open shader " << path << std::endl;
        return 0;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    std::string src = ss.str();
    const char* cstr = src.c_str();

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &cstr, nullptr);
    glCompileShader(shader);

    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error (" << path << "):\n" << log << std::endl;
    }
    return shader;
}

GLuint GLDisplay::linkProgram(GLuint vert, GLuint frag) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);

    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::cerr << "Shader link error:\n" << log << std::endl;
    }
    return prog;
}

} // namespace pt
