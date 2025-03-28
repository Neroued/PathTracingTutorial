#pragma once

#include "config.h"

#include <Material.h>
#include <Primitive.h>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <cuda_runtime.h>
#include <vector>

BEGIN_NAMESPACE_PT

class Scene : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core {
public:
    Scene(QWidget* parent = nullptr);
    ~Scene();

    // void loadScene(); // 加载场景

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

private:
    void compileShaders();   // 编译所需的着色器

    void computePass();      // 计算pass
    void mixPass();
    void renderShaderPass(); // 渲染pass，展示计算结果的pass

    // void uploadScene();                                                      // 上传场景信息

    void initializeQuad();                                                   // 初始化全屏四边形
    void createTexture(GLuint* texture, int width, int height, GLuint unit); // 创建材质

    void checkCudaErrors(cudaError_t err, const char* msg);
    void initCudaConstants();

private:
    QOpenGLShaderProgram* m_renderProgram; // 展示结果的着色器

    QOpenGLVertexArrayObject m_screenVAO;  // 用于屏幕的vao和vbo
    QOpenGLBuffer m_screenVBO;

    GLuint m_computeTexture;                 // 存放计算结果的材质
    GLuint m_imageTexture;                   // 存储历史结果的材质

    cudaGraphicsResource* m_computeResource; // 对应 cuda 的计算存储资源
    cudaGraphicsResource* m_imageResource;   // 对应历史结果的资源

    int m_width, m_height;
    GLuint m_frameCount = 0;

    std::vector<Primitive> m_primitives;
};

END_NAMESPACE_PT