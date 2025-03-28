#pragma once

#include <cuda_gl_interop.h>
#include <driver_types.h>
#include <Material.h>
#include <Primitive.h>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <vector>

class ptScene : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core {
public:
    ptScene(QWidget* parent = nullptr);
    ~ptScene();

    void loadScene(); // 加载场景

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

private:
    void compileShaders();                                                   // 编译所需的着色器

    void cudaPass();                                                         // 计算pass
    void computeShaderPass();                                                // 计算pass，进行路径追踪的pass
    void mixShaderPass();                                                    // 混合pass，将新的结果与之前的混合
    void renderShaderPass();                                                 // 渲染pass，展示计算结果的pass

    void uploadScene();                                                      // 上传场景信息

    void initializeQuad();                                                   // 初始化全屏四边形
    void createTexture(GLuint* texture, int width, int height, GLuint unit); // 创建材质

    void checkCudaErrors(cudaError_t err, const char* msg);

private:
    QOpenGLShaderProgram* m_computeProgram; // 计算着色器
    QOpenGLShaderProgram* m_mixProgram;     // 混合着色器
    QOpenGLShaderProgram* m_renderProgram;  // 展示结果的着色器

    QOpenGLVertexArrayObject m_screenVAO;   // 用于屏幕的vao和vbo
    QOpenGLBuffer m_screenVBO;

    GLuint m_computeTexture; // 存放计算结果的材质
    GLuint m_imageTexture;   // 存储历史结果的材质
    GLuint m_cudaTexture;    // cuda 计算所用的材质

    cudaGraphicsResource* m_cudaResource;

    int m_width, m_height;
    GLuint m_frameCount = 0;

    std::vector<Primitive> m_primitives;
};