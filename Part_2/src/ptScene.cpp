#include <ptScene.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <QDebug>
#include <surface_types.h>
#include <vector_types.h>

ptScene::ptScene(QWidget* parent)
    : QOpenGLWidget(parent), m_renderProgram(nullptr), m_computeTexture(0), m_imageTexture(0), m_computeResource(nullptr), m_imageResource(nullptr),
      m_screenVBO(QOpenGLBuffer::VertexBuffer), m_width(800), m_height(800) {
    resize(m_width, m_height);
}

ptScene::~ptScene() {
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
        checkCudaErrors(cudaGraphicsUnregisterResource(m_computeResource), "cudaGraphicsUnregisterResource");
        m_computeResource = nullptr;
    }

    if (m_imageResource) {
        checkCudaErrors(cudaGraphicsUnregisterResource(m_imageResource), "cudaGraphicsUnregisterResource");
        m_imageResource = nullptr;
    }

    doneCurrent();
}

void ptScene::initializeGL() {
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
    createTexture(&m_computeTexture, m_width, m_height, 0); // binding = 0
    createTexture(&m_imageTexture, m_width, m_height, 1);   // binding = 1

    // 将纹理注册到 cuda
    checkCudaErrors(cudaGraphicsGLRegisterImage(&m_computeResource, m_computeTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore),
                    "cudaGraphicsGLRegisterImage");

    checkCudaErrors(cudaGraphicsGLRegisterImage(&m_imageResource, m_imageTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore),
                    "cudaGraphicsGLRegisterImage");

    // -------------------------------
    // 构建全屏四边形（屏幕四边形）的 VAO/VBO
    // -------------------------------
    initializeQuad();
}

void ptScene::resizeGL(int w, int h) {
    m_width  = w;
    m_height = h;

    if (m_computeTexture) {
        glDeleteTextures(1, &m_computeTexture);
        m_computeTexture = 0;
    }
    if (m_imageTexture) {
        glDeleteTextures(1, &m_imageTexture);
        m_imageTexture = 0;
    }

    // 生成新的纹理
    createTexture(&m_computeTexture, m_width, m_height, 0);
    createTexture(&m_imageTexture, m_width, m_height, 1);

    if (m_computeResource) {
        checkCudaErrors(cudaGraphicsUnregisterResource(m_computeResource), "cudaGraphicsUnregisterResource");
        m_computeResource = nullptr;
    }
    if (m_imageResource) {
        checkCudaErrors(cudaGraphicsUnregisterResource(m_imageResource), "cudaGraphicsUnregisterResource");
        m_imageResource = nullptr;
    }

    // 将纹理注册到 cuda
    checkCudaErrors(cudaGraphicsGLRegisterImage(&m_computeResource, m_computeTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore),
                    "cudaGraphicsGLRegisterImage");

    checkCudaErrors(cudaGraphicsGLRegisterImage(&m_imageResource, m_imageTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore),
                    "cudaGraphicsGLRegisterImage");
    // 重置帧计数
    m_frameCount = 0;

    glViewport(0, 0, w, h);
}

void ptScene::paintGL() {
    // -------------------------------
    // 1. 调度计算着色器更新纹理
    // -------------------------------
    computePass();

    // -------------------------
    // 2. 将新计算结果与之前的混合
    // -------------------------
    mixPass();

    // -------------------------------
    // 3. 渲染全屏四边形，显示计算结果纹理
    // -------------------------------
    renderShaderPass();

    // -----------------------
    // 4. 调用 update() 触发重绘
    // -----------------------
    m_frameCount++;
    update();
}

void ptScene::compileShaders() {
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

void ptScene::createTexture(GLuint* texture, int width, int height, GLuint unit) {
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

// pathTracingKernel.cu 提供的接口, 调用 __global__ 函数
extern "C" void launchKernel(cudaSurfaceObject_t surface, int width, int height);

void ptScene::computePass() {
    // 将 OpenGL 的纹理资源映射到 CUDA
    checkCudaErrors(cudaGraphicsMapResources(1, &m_computeResource, 0), "cudaGraphicsMapResources");

    // 获取映射后的 cudaArray
    cudaArray_t textureArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&textureArray, m_computeResource, 0, 0), "cudaGraphicsSubResourceGetMappedArray");

    // 创建资源描述符
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    // 创建 CUDA surface 对象
    cudaSurfaceObject_t surfaceObj = 0;
    checkCudaErrors(cudaCreateSurfaceObject(&surfaceObj, &resDesc), "cudaCreateSurfaceObject");

    // 启动 CUDA kernel
    launchKernel(surfaceObj, m_width, m_height);

    // 销毁 surface 对象
    checkCudaErrors(cudaDestroySurfaceObject(surfaceObj), "cudaDestroySurfaceObject");

    // 解除资源映射
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_computeResource, 0), "cudaGraphicsUnmapResources");
}

extern "C" void launchMixKernel(cudaSurfaceObject_t surfaceNew, cudaSurfaceObject_t surfaceAcc, int width, int height, unsigned int sampleCount);

void ptScene::mixPass() {
    // 将 OpenGL 的纹理资源映射到 CUDA
    checkCudaErrors(cudaGraphicsMapResources(1, &m_computeResource, 0), "cudaGraphicsMapResources");
    checkCudaErrors(cudaGraphicsMapResources(1, &m_imageResource, 0), "cudaGraphicsMapResources");

    // 获取映射后的 cudaArray
    cudaArray_t computeArray, imageAray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&computeArray, m_computeResource, 0, 0), "cudaGraphicsSubResourceGetMappedArray");
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&imageAray, m_imageResource, 0, 0), "cudaGraphicsSubResourceGetMappedArray");

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
    checkCudaErrors(cudaCreateSurfaceObject(&computeSurfaceObj, &computeResDesc), "cudaCreateSurfaceObject");
    checkCudaErrors(cudaCreateSurfaceObject(&imageSurfaceObj, &imageResDesc), "cudaCreateSurfaceObject");

    // 启动 CUDA kernel
    launchMixKernel(computeSurfaceObj, imageSurfaceObj, m_width, m_height, m_frameCount + 1);

    // 销毁 surface 对象
    checkCudaErrors(cudaDestroySurfaceObject(computeSurfaceObj), "cudaDestroySurfaceObject");
    checkCudaErrors(cudaDestroySurfaceObject(imageSurfaceObj), "cudaDestroySurfaceObject");

    // 解除资源映射
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_computeResource, 0), "cudaGraphicsUnmapResources");
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_imageResource, 0), "cudaGraphicsUnmapResources");
}

void ptScene::renderShaderPass() {
    // 使用着色器绘制全屏四边形并采样compute shader计算的texture
    m_renderProgram->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    m_screenVAO.bind();

    // 激活并绑定纹理到纹理单元 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_imageTexture);
    // 将 uniform "imageTexture" 绑定到纹理单元 0
    m_renderProgram->setUniformValue("imageTexture", 0);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    m_screenVAO.release();
    m_renderProgram->release();
}

void ptScene::initializeQuad() {
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

void ptScene::checkCudaErrors(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        qFatal() << "CUDA Error:" << msg << "\n    Code:" << static_cast<int>(err) << "\n    Name:" << cudaGetErrorName(err)
                 << "\n    Description:" << cudaGetErrorString(err);
    }
}