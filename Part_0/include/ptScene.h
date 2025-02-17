#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>

class ptScene : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core
{
public:
    ptScene(QWidget *parent = nullptr);
    ~ptScene();

    void loadScene(); // 加载场景

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

private:
    void compileShaders(); // 编译所需的着色器

    void computeShaderPass(); // 计算pass，即进行路径追踪的pass
    void renderShaderPass();  // 渲染pass，即展示计算结果的pass

    void uploadScene(); // 上传场景信息

    void initializeQuad(); // 初始化全屏四边形
    void createTexture(GLuint *texture, int width, int height, GLuint unit); // 创建材质

private:
    QOpenGLShaderProgram *m_computeProgram; // 计算着色器
    QOpenGLShaderProgram *m_renderProgram;  // 展示结果的着色器

    QOpenGLVertexArrayObject m_screenVAO; // 用于屏幕的vao和vbo
    QOpenGLBuffer m_screenVBO;

    GLuint m_computeTexture; // 存放计算结果的材质

    int m_width, m_height;
};