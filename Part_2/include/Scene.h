#pragma once

#include "SceneData.h"
#include "Triangle.h"
#include "config.h"
#include "vec3.h"
#include "mat4.h"
#include "BVH.h"

#include <Material.h>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

BEGIN_NAMESPACE_PT

class Scene : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core {
public:
    Scene(QWidget* parent = nullptr);
    ~Scene();

    void loadScene(); // 加载场景

    // 添加obj文件
    void addObj(const std::string& filename, int materialID, const mat4& transform = mat4::identity());

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

private:
    void compileShaders();                                                   // 编译所需的着色器

    void computePass();                                                      // 计算pass
    void mixPass();
    void renderShaderPass();                                                 // 渲染pass，展示计算结果

    void uploadScene();                                                      // 上传场景信息至 GPU

    void initializeQuad();                                                   // 初始化全屏四边形
    void createTexture(GLuint* texture, int width, int height, GLuint unit); // 创建材质

    void initSceneConstants();

    void showFPS();
    void cleanup(); // 清除 cuda 分配的资源

    // 辅助函数，用来添加方块
    void addCube(const vec3& minCorner, const vec3& maxCorner, int materialID, const mat4& transform = mat4::identity());

private:
    QOpenGLShaderProgram* m_renderProgram; // 展示结果的着色器

    QOpenGLVertexArrayObject m_screenVAO;  // 用于屏幕的vao和vbo
    QOpenGLBuffer m_screenVBO;

    GLuint m_computeTexture;                 // 存放计算结果的材质
    GLuint m_imageTexture;                   // 存储历史结果的材质

    cudaGraphicsResource* m_computeResource; // 对应 cuda 的计算存储资源
    cudaGraphicsResource* m_imageResource;   // 对应历史结果的资源

    SceneData m_sceneData;

    // 用于显示fps
    std::chrono::steady_clock::time_point m_start;     // 程序累计运行时间
    std::chrono::steady_clock::time_point m_lastStart; // 上一次显示之后的运行时间
    GLuint m_elapsedFrameCount = 0;

    std::vector<Triangle> m_triangles;
    std::vector<Material> m_materials;

    BVH m_bvh;
};

END_NAMESPACE_PT