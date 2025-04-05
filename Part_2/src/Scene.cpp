#include <Scene.h>
#include <chrono>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <QDebug>
#include <cudaKernelFunctions.h>
#include <iostream>
#include <iomanip>
#include <qlogging.h>
#include <qopenglext.h>
#include <qstringtokenizer.h>
#include "BVH.h"
#include "Material.h"
#include "SceneConstants.cuh"
#include "Triangle.h"
#include "config.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "SceneConfig.h"
#include "Texture.h"
#include "ImageLoader.h"

#include "mat4.h"
#include "tiny_obj_loader.h"

BEGIN_NAMESPACE_PT

Scene::Scene(QWidget* parent)
    : QOpenGLWidget(parent), m_renderProgram(nullptr), m_computeTexture(0), m_imageTexture(0), m_computeResource(nullptr), m_imageResource(nullptr),
      m_screenVBO(QOpenGLBuffer::VertexBuffer) {
    m_sceneData.width  = WIDTH;
    m_sceneData.height = HEIGHT;
    resize(m_sceneData.width, m_sceneData.height);
}

Scene::~Scene() {
    makeCurrent();

    m_screenVAO.destroy();
    m_screenVBO.destroy();
    if (m_computeTexture) {
        glDeleteTextures(1, &m_computeTexture);
        m_computeTexture = 0;
    }
    if (m_imageTexture) {
        glDeleteTextures(1, &m_imageTexture);
        m_imageTexture = 0;
    }

    delete m_renderProgram;

    if (m_computeResource) {
        CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(m_computeResource));
        m_computeResource = nullptr;
    }

    if (m_imageResource) {
        CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(m_imageResource));
        m_imageResource = nullptr;
    }

    doneCurrent();

    cleanup();
}

void Scene::initializeGL() {
    // -----------------
    // 初始化 OpenGL 函数
    // -----------------
    initializeOpenGLFunctions();

    // -----------------
    // 编译计算着色器程序
    // -----------------
    compileShaders();

    // ------------------------
    // 创建用于存储计算结果的纹理
    // ------------------------
    createTexture(&m_computeTexture, m_sceneData.width, m_sceneData.height, 0); // binding = 0
    createTexture(&m_imageTexture, m_sceneData.width, m_sceneData.height, 1);   // binding = 1

    // 将纹理注册到 cuda
    CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&m_computeResource, m_computeTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&m_imageResource, m_imageTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    // -------------------------------
    // 构建全屏四边形（屏幕四边形）的 VAO/VBO
    // -------------------------------
    initializeQuad();

    // -----------
    // 记录开始时间
    // -----------
    m_lastStart = m_start = std::chrono::high_resolution_clock::now();

    // --------------------
    // 初始化 SceneConstants
    // --------------------
    initSceneConstants();
}

void Scene::resizeGL(int w, int h) {
    m_sceneData.width  = w;
    m_sceneData.height = h;

    if (m_computeTexture) {
        glDeleteTextures(1, &m_computeTexture);
        m_computeTexture = 0;
    }
    if (m_imageTexture) {
        glDeleteTextures(1, &m_imageTexture);
        m_imageTexture = 0;
    }

    // 生成新的纹理
    createTexture(&m_computeTexture, m_sceneData.width, m_sceneData.height, 0);
    createTexture(&m_imageTexture, m_sceneData.width, m_sceneData.height, 1);

    if (m_computeResource) {
        CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(m_computeResource));
        m_computeResource = nullptr;
    }
    if (m_imageResource) {
        CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(m_imageResource));
        m_imageResource = nullptr;
    }

    // 将纹理注册到 cuda
    CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&m_computeResource, m_computeTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&m_imageResource, m_imageTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    // 重置帧计数
    m_sceneData.frameCount = 0;

    glViewport(0, 0, w, h);
}

void Scene::paintGL() {
    if (m_sceneData.frameCount * SAMPLE_PER_FRAME < TARGET_SAMPLES) {
        // -------------------------------
        // 1. 调度计算核心更新纹理
        // -------------------------------
        computePass();

        // -------------------------
        // 2. 将新计算结果与之前的混合
        // -------------------------
        mixPass();
    }

    if (m_sceneData.frameCount * SAMPLE_PER_FRAME >= TARGET_SAMPLES && (m_sceneData.frameCount - 1) * SAMPLE_PER_FRAME < TARGET_SAMPLES) {
        saveImage();
        std::cout << std::endl;
    }

    // -------------------------------
    // 3. 渲染全屏四边形，显示计算结果纹理
    // -------------------------------
    renderShaderPass();

    // -----------------------
    // 4. 显示fps
    // -----------------------
    m_sceneData.frameCount++;
    m_elapsedFrameCount++;
    showFPS();

    // -----------------------
    // 5. 调用 update() 触发重绘
    // -----------------------
    update();
}

void Scene::compileShaders() {
    // ----------------------
    // 编译顶点与片段着色器程序
    // ----------------------
    m_renderProgram = new QOpenGLShaderProgram();
    if (!m_renderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "E:\\code\\c++\\PathTracingTutorial\\Part_2\\shaders\\pt_vertex.glsl")) {
        qWarning() << "Failed to compile vertex shader:" << m_renderProgram->log();
        return;
    }
    if (!m_renderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "E:\\code\\c++\\PathTracingTutorial\\Part_2\\shaders\\pt_fragment.glsl")) {
        qWarning() << "Failed to compile fragment shader:" << m_renderProgram->log();
        return;
    }
    if (!m_renderProgram->link()) {
        qWarning() << "Failed to link render shader program:" << m_renderProgram->log();
        return;
    }

    qDebug() << "Successfully compiled shaders";
}

void Scene::createTexture(GLuint* texture, int width, int height, GLuint unit) {
    if (*texture) {
        glDeleteTextures(1, texture);
        *texture = 0;
    }

    // 生成新的纹理
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);

    // 设置纹理参数
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // 分配纹理内存，使用 GL_RGBA32F 格式，高精度浮点数
    // 参数解释：
    //   - target: GL_TEXTURE_2D 表示创建2D纹理
    //   - level: 0 为基本纹理层
    //   - internalFormat: GL_RGBA32F 表示每个像素存储 RGBA 四个通道的 32 位浮点数
    //   - width, height: 根据窗口尺寸设置纹理尺寸
    //   - border: 必须为 0
    //   - format: GL_RGBA 表示数据格式为 RGBA
    //   - type: GL_FLOAT 表示数据类型为浮点型
    //   - data: NULL 表示不传入初始数据
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

    // 将新纹理绑定到图像单元 unit
    glBindImageTexture(unit, *texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
}

void Scene::computePass() {
    // 将 OpenGL 的纹理资源映射到 CUDA
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_computeResource, 0));

    // 获取映射后的 cudaArray
    cudaArray_t textureArray;
    CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&textureArray, m_computeResource, 0, 0));

    // 创建资源描述符
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    // 创建 CUDA surface 对象
    cudaSurfaceObject_t surfaceObj = 0;
    CUDA_SAFE_CALL(cudaCreateSurfaceObject(&surfaceObj, &resDesc));

    // 启动 CUDA kernel
    launchKernel(surfaceObj, std::move(m_sceneData));

    // 销毁 surface 对象
    CUDA_SAFE_CALL(cudaDestroySurfaceObject(surfaceObj));

    // 解除资源映射
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &m_computeResource, 0));
}

void Scene::mixPass() {
    // 将 OpenGL 的纹理资源映射到 CUDA
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_computeResource, 0));
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_imageResource, 0));

    // 获取映射后的 cudaArray
    cudaArray_t computeArray, imageAray;
    CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&computeArray, m_computeResource, 0, 0));
    CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&imageAray, m_imageResource, 0, 0));

    // 创建资源描述符
    cudaResourceDesc computeResDesc, imageResDesc;
    memset(&computeResDesc, 0, sizeof(computeResDesc));
    memset(&imageResDesc, 0, sizeof(imageResDesc));
    computeResDesc.resType = imageResDesc.resType = cudaResourceTypeArray;
    computeResDesc.res.array.array                = computeArray;
    imageResDesc.res.array.array                  = imageAray;

    // 创建 CUDA surface 对象
    cudaSurfaceObject_t computeSurfaceObj = 0;
    cudaSurfaceObject_t imageSurfaceObj   = 0;
    CUDA_SAFE_CALL(cudaCreateSurfaceObject(&computeSurfaceObj, &computeResDesc));
    CUDA_SAFE_CALL(cudaCreateSurfaceObject(&imageSurfaceObj, &imageResDesc));

    // 启动 CUDA kernel
    launchMixKernel(computeSurfaceObj, imageSurfaceObj, m_sceneData);

    // 销毁 surface 对象
    CUDA_SAFE_CALL(cudaDestroySurfaceObject(computeSurfaceObj));
    CUDA_SAFE_CALL(cudaDestroySurfaceObject(imageSurfaceObj));

    // 解除资源映射
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &m_computeResource, 0));
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &m_imageResource, 0));
}

void Scene::renderShaderPass() {
    // 使用着色器绘制全屏四边形并采样compute shader计算的texture
    m_renderProgram->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    m_screenVAO.bind();

    // 激活并绑定纹理到纹理单元 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_imageTexture);
    // 将 uniform "imageTexture" 绑定到纹理单元 0
    m_renderProgram->setUniformValue("imageTexture", 0);

    m_renderProgram->setUniformValue("gamma", m_gamma);
    m_renderProgram->setUniformValue("exposure", m_exposure);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    m_screenVAO.release();
    m_renderProgram->release();
}

void Scene::initializeQuad() {
    // 定义全屏四边形的顶点数据（6 个顶点，每个顶点包含 2 个位置与 2 个纹理坐标）
    float quadVertices[] = {
        // 位置      // 纹理坐标
        -1.0f, 1.0f,  0.0f, 1.0f, // 左上
        -1.0f, -1.0f, 0.0f, 0.0f, // 左下
        1.0f,  -1.0f, 1.0f, 0.0f, // 右下

        -1.0f, 1.0f,  0.0f, 1.0f, // 左上
        1.0f,  -1.0f, 1.0f, 0.0f, // 右下
        1.0f,  1.0f,  1.0f, 1.0f  // 右上
    };

    // 初始化 VAO
    m_screenVAO.create();
    m_screenVAO.bind();

    // 初始化 VBO
    m_screenVBO.create();
    m_screenVBO.bind();
    m_screenVBO.allocate(quadVertices, sizeof(quadVertices));

    // 设置顶点属性
    // 属性 0：位置（2 个浮点数），偏移 0，步长为 4*sizeof(float)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
    // 属性 1：纹理坐标（2 个浮点数），偏移为 2*sizeof(float)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    m_screenVAO.release();
}

void Scene::showFPS() {
    using namespace std::chrono;
    auto elapsed     = high_resolution_clock::now() - m_lastStart;
    auto duration_ms = duration_cast<milliseconds>(elapsed);
    if (m_elapsedFrameCount == 10 || duration_ms.count() >= 100) {
        float fps              = m_elapsedFrameCount * 1000.0f / static_cast<float>(duration_ms.count());
        auto total_duration_ms = duration_cast<milliseconds>(high_resolution_clock::now() - m_start);
        std::cout << std::fixed << std::setprecision(2) << "\rFPS: " << fps << " Elapsed time: " << static_cast<float>(duration_ms.count()) / 1000.0f
                  << "s" << " Total frames: " << m_sceneData.frameCount << " Total time: " << static_cast<float>(total_duration_ms.count()) / 1000.0f
                  << "s     " << std::flush;
        m_elapsedFrameCount = 0;
        m_lastStart         = high_resolution_clock::now();
    }
}

void Scene::initSceneConstants() {
    SceneConstants host;
    host.width          = m_sceneData.width;
    host.height         = m_sceneData.height;
    host.cameraPos      = CAMERA_POS;
    host.screenZ        = SCREEN_Z;
    host.samplePerFrame = SAMPLE_PER_FRAME;
    host.targetSamples  = TARGET_SAMPLES;
    host.depth          = DEPTH;
    uploadSceneConstant(host);
}

void Scene::uploadScene() {
    CUDA_SAFE_CALL(cudaMalloc(&m_sceneData.triangles, m_triangles.size() * sizeof(Triangle)));
    CUDA_SAFE_CALL(cudaMemcpy(m_sceneData.triangles, m_triangles.data(), m_triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    m_sceneData.numTriangles = m_triangles.size();

    CUDA_SAFE_CALL(cudaMalloc(&m_sceneData.materials, m_materials.size() * sizeof(Material)));
    CUDA_SAFE_CALL(cudaMemcpy(m_sceneData.materials, m_materials.data(), m_materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    m_sceneData.numMaterials = m_materials.size();

    CUDA_SAFE_CALL(cudaMalloc(&m_sceneData.bvhNodes, m_bvh.nodes.size() * sizeof(BVHNode)));
    CUDA_SAFE_CALL(cudaMemcpy(m_sceneData.bvhNodes, m_bvh.nodes.data(), m_bvh.nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));
    m_sceneData.numBvhNodes = m_bvh.nodes.size();

    m_sceneData.hdrTex = m_hdrTex.cuTexture;
}

void Scene::cleanup() {
    if (m_sceneData.triangles) CUDA_SAFE_CALL(cudaFree(m_sceneData.triangles));
    if (m_sceneData.materials) CUDA_SAFE_CALL(cudaFree(m_sceneData.materials));
    if (m_sceneData.bvhNodes) CUDA_SAFE_CALL(cudaFree(m_sceneData.bvhNodes));
    m_sceneData.triangles = nullptr;
    m_sceneData.materials = nullptr;
    m_sceneData.bvhNodes  = nullptr;
}

void Scene::loadScene() {
    cleanup();
    m_triangles.clear();
    m_materials.clear();

    const vec3 RED(1, 0.5, 0.5);
    const vec3 GREEN(0.5, 1, 0.5);
    const vec3 BLUE(0.5, 0.5, 1);
    const vec3 YELLOW(1.0, 1.0, 0.1);
    const vec3 CYAN(0.1, 1.0, 1.0);
    const vec3 MAGENTA(1.0, 0.1, 1.0);
    const vec3 GRAY(0.5, 0.5, 0.5);
    const vec3 WHITE(1, 1, 1);
    const vec3 BLACK(0.0, 0.0, 0.0);

    // 光源
    Triangle l1   = Triangle(vec3(0.4, 0.99, 0.4), vec3(-0.4, 0.99, -0.4), vec3(-0.4, 0.99, 0.4));
    Triangle l2   = Triangle(vec3(0.4, 0.99, 0.4), vec3(0.4, 0.99, -0.4), vec3(-0.4, 0.99, -0.4));
    l1.materialID = 0;
    l2.materialID = 0;
    Material light;
    light.emissive  = 1.0f;
    light.baseColor = WHITE;
    // m_triangles.push_back(l1);
    // m_triangles.push_back(l2);
    m_materials.push_back(light);

    // 背景盒子
    // bottom
    Triangle bottom1   = Triangle(vec3(1, -1, 1), vec3(-1, -1, -1), vec3(-1, -1, 1));
    Triangle bottom2   = Triangle(vec3(1, -1, 1), vec3(1, -1, -1), vec3(-1, -1, -1));
    bottom1.materialID = 1;
    bottom2.materialID = 1;
    Material bottom;
    bottom.baseColor = WHITE;
    m_triangles.push_back(bottom1);
    m_triangles.push_back(bottom2);
    m_materials.push_back(bottom);

    // top
    Triangle top1   = Triangle(vec3(1, 1, 1), vec3(-1, 1, 1), vec3(-1, 1, -1));
    Triangle top2   = Triangle(vec3(1, 1, 1), vec3(-1, 1, -1), vec3(1, 1, -1));
    top1.materialID = 2;
    top2.materialID = 2;
    Material top;
    top.baseColor = WHITE;
    m_triangles.push_back(top1);
    m_triangles.push_back(top2);
    m_materials.push_back(top);

    // back
    Triangle back1   = Triangle(vec3(1, -1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1));
    Triangle back2   = Triangle(vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1));
    back1.materialID = 3;
    back2.materialID = 3;
    Material back;
    back.baseColor = CYAN;
    m_triangles.push_back(back1);
    m_triangles.push_back(back2);
    m_materials.push_back(back);

    // left
    Triangle left1   = Triangle(vec3(-1, -1, -1), vec3(-1, 1, 1), vec3(-1, -1, 1));
    Triangle left2   = Triangle(vec3(-1, -1, -1), vec3(-1, 1, -1), vec3(-1, 1, 1));
    left1.materialID = 4;
    left2.materialID = 4;
    Material left;
    left.baseColor = BLUE;
    m_triangles.push_back(left1);
    m_triangles.push_back(left2);
    m_materials.push_back(left);

    // right
    Triangle right1   = Triangle(vec3(1, 1, 1), vec3(1, -1, -1), vec3(1, -1, 1));
    Triangle right2   = Triangle(vec3(1, -1, -1), vec3(1, 1, 1), vec3(1, 1, -1));
    right1.materialID = 5;
    right2.materialID = 5;
    Material right;
    right.baseColor = RED;
    m_triangles.push_back(right1);
    m_triangles.push_back(right2);
    m_materials.push_back(right);

    // 添加立方体1（较大）
    Material cube1Mat;
    cube1Mat.baseColor = GREEN; // 使用预定义的绿色
    m_materials.push_back(cube1Mat);
    int cube1MaterialID = m_materials.size() - 1;
    vec3 cube1Min(-0.7, -1, -0.6);
    vec3 cube1Max(-0.1, 0.2, 0.0);
    // 示例：对“较大立方体”绕 Y 轴旋转
    mat4 transform1;
    vec3 cube1Center = 0.5f * (cube1Min + cube1Max);
    // 第1步: 将立方体中心移到原点
    transform1.translate(cube1Center);
    // 第2步: 旋转
    float angle1 = 30.0f; // 例如旋转 30°
    transform1.rotate(angle1, {0.0f, 1.0f, 0.0f});
    // 第3步: 平移回去
    transform1.translate(-cube1Center);
    // 最后调用 addCube() 时，把该 transform 传进去
    // addCube(cube1Min, cube1Max, cube1MaterialID, transform1);

    // 添加立方体2（较小）
    Material cube2Mat;
    cube2Mat.baseColor = YELLOW; // 使用预定义的黄色
    m_materials.push_back(cube2Mat);
    int cube2MaterialID = m_materials.size() - 1;
    vec3 cube2Min(0.2, -1, -0.2);
    vec3 cube2Max(0.8, -0.4, 0.5);
    // addCube(cube2Min, cube2Max, cube2MaterialID);

    // 镜面反射材质
    Material mirrorMat;
    mirrorMat.baseColor    = YELLOW;
    mirrorMat.specularRate = 0.9f;
    mirrorMat.roughness    = 0.3f;
    m_materials.push_back(mirrorMat);
    int mirrorMaterialID = m_materials.size() - 1;

    // 在右边添加一个平片镜子
    Triangle mirror1   = Triangle(vec3(1, -1, -0.7), vec3(1, 1, -0.7), vec3(0.5, 1, -1));  // 右下， 右上， 左上
    Triangle mirror2   = Triangle(vec3(1, -1, -0.7), vec3(0.5, 1, -1), vec3(0.5, -1, -1)); // 右下， 左上， 左下
    mirror1.materialID = mirrorMaterialID;
    mirror2.materialID = mirrorMaterialID;
    // m_triangles.push_back(mirror1);
    // m_triangles.push_back(mirror2);

    // 加载一个茶杯
    mat4 tranform1 = mat4::translation(0.0f, -1.0f, 0.0f) * mat4::scaling(0.01f, 0.01f, 0.01f);
    // addObj("E:\\code\\c++\\PathTracingTutorial\\Part_2\\models\\teapot.obj", mirrorMaterialID, tranform1);

    // 加载 bunny
    mat4 tranform2 = mat4::translation(0.5f, -0.48f, 0.15f) * mat4::scaling(2.5f, 2.5f, 2.5f);
    // addObj("E:\\code\\c++\\PathTracingTutorial\\Part_2\\models\\Stanford Bunny.obj", 2, tranform2);

    // 加载 dragon
    float angleRad  = PI / 2.0f;
    float scale     = 2.0f;
    mat4 transform3 = mat4::translation(0.0f, -0.5f, 0.0f) * mat4::rotation(angleRad, vec3(0.0f, 1.0f, 0.0f)) * mat4::scaling(scale, scale, scale);
    addObj("E:\\code\\c++\\PathTracingTutorial\\Part_2\\models\\dragon.obj", mirrorMaterialID, transform3);

    // 构建 BVH
    m_bvh.build(m_triangles, 0, m_triangles.size());

    std::cout << "Triangles: " << m_triangles.size();
    std::cout << " BVH nodes: " << m_bvh.nodes.size() << std::endl;

    // 加载 hdr 贴图
    Image img;
    ImageLoader::load(img, "E:\\code\\c++\\PathTracingTutorial\\Part_2\\models\\brown_photostudio_02_4k.hdr");

    m_hdrTex = Texture(img);

    uploadScene();
}

void Scene::addCube(const vec3& minCorner, const vec3& maxCorner, int materialID, const mat4& transform) {
    // 构造立方体八个角点
    vec3 v000(minCorner.x, minCorner.y, minCorner.z);
    vec3 v001(minCorner.x, minCorner.y, maxCorner.z);
    vec3 v010(minCorner.x, maxCorner.y, minCorner.z);
    vec3 v011(minCorner.x, maxCorner.y, maxCorner.z);
    vec3 v100(maxCorner.x, minCorner.y, minCorner.z);
    vec3 v101(maxCorner.x, minCorner.y, maxCorner.z);
    vec3 v110(maxCorner.x, maxCorner.y, minCorner.z);
    vec3 v111(maxCorner.x, maxCorner.y, maxCorner.z);

    // 应用变换（内部调用 operator*(vec3) 实现点变换，w默认为1）
    v000 = transform * v000;
    v001 = transform * v001;
    v010 = transform * v010;
    v011 = transform * v011;
    v100 = transform * v100;
    v101 = transform * v101;
    v110 = transform * v110;
    v111 = transform * v111;

    // 底面 (y = minCorner.y)
    Triangle tri1(v000, v100, v101);
    Triangle tri2(v000, v101, v001);
    tri1.materialID = materialID;
    tri2.materialID = materialID;
    m_triangles.push_back(tri1);
    m_triangles.push_back(tri2);

    // 顶面 (y = maxCorner.y)
    Triangle tri3(v010, v011, v111);
    Triangle tri4(v010, v111, v110);
    tri3.materialID = materialID;
    tri4.materialID = materialID;
    m_triangles.push_back(tri3);
    m_triangles.push_back(tri4);

    // 前面 (z = maxCorner.z)
    Triangle tri5(v001, v101, v111);
    Triangle tri6(v001, v111, v011);
    tri5.materialID = materialID;
    tri6.materialID = materialID;
    m_triangles.push_back(tri5);
    m_triangles.push_back(tri6);

    // 后面 (z = minCorner.z)
    Triangle tri7(v000, v010, v110);
    Triangle tri8(v000, v110, v100);
    tri7.materialID = materialID;
    tri8.materialID = materialID;
    m_triangles.push_back(tri7);
    m_triangles.push_back(tri8);

    // 左面 (x = minCorner.x)
    Triangle tri9(v000, v001, v011);
    Triangle tri10(v000, v011, v010);
    tri9.materialID  = materialID;
    tri10.materialID = materialID;
    m_triangles.push_back(tri9);
    m_triangles.push_back(tri10);

    // 右面 (x = maxCorner.x)
    Triangle tri11(v100, v110, v111);
    Triangle tri12(v100, v111, v101);
    tri11.materialID = materialID;
    tri12.materialID = materialID;
    m_triangles.push_back(tri11);
    m_triangles.push_back(tri12);
}

void Scene::addObj(const std::string& filename, int materialID, const mat4& transform) {
    tinyobj::ObjReaderConfig readerConfig;
    readerConfig.mtl_search_path = "./";

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename, readerConfig)) {
        if (!reader.Error().empty()) { std::cerr << "TinyObjReader: " << reader.Error(); }
        return;
    }

    if (!reader.Warning().empty()) { std::cerr << "TinyObjReader: " << reader.Warning(); }

    const auto& attrib    = reader.GetAttrib();
    const auto& shapes    = reader.GetShapes();
    const auto& materials = reader.GetMaterials();

    // 计算 tranform 的逆转置用于变换法线
    const mat4 transform_inv_T = transform.inverse().transposed();

    for (size_t s = 0; s < shapes.size(); ++s) {
        size_t indexOffset = 0;
        m_triangles.reserve(m_triangles.size() + shapes[s].mesh.num_face_vertices.size());
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
            size_t fv = shapes[s].mesh.num_face_vertices[f]; // 面的顶点数量，应当为 3
            if (fv == 3) {                                   // 读取三个顶点，存放到三个 vec3 中
                vec3 p[3], n[3];
                bool haveNormal = true;
                for (size_t v = 0; v < fv; ++v) {
                    tinyobj::index_t idx = shapes[s].mesh.indices[indexOffset + v];

                    tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                    p[v]               = transform.transformPoint(vec3(vx, vy, vz));

                    if (idx.normal_index >= 0 && haveNormal) {
                        tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                        tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                        tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                        n[v]               = transform_inv_T.transformVector(vec3(nx, ny, nz));
                    } else {
                        haveNormal = false;
                    }
                }

                if (haveNormal) {
                    Triangle& tri  = m_triangles.emplace_back(p[0], p[1], p[2], n[0], n[1], n[2]);
                    tri.materialID = materialID;
                } else {
                    Triangle& tri  = m_triangles.emplace_back(p[0], p[1], p[2]);
                    tri.materialID = materialID;
                }
            }

            indexOffset += fv;
        }
    }
}

void Scene::saveImage() {
    Image img;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_imageTexture);

    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &img.width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &img.height);
    img.channel = 4;

    img.data.resize(img.width * img.height * img.channel);

    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, img.data.data()); // 读取纹理内容

    ImageLoader::write(img, "default.hdr");
}

END_NAMESPACE_PT