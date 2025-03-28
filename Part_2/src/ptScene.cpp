#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <GL/gl.h>
#include <ptScene.h>
#include <QDebug>
#include <qlogging.h>
#include <surface_types.h>
#include <vector_types.h>

ptScene::ptScene(QWidget* parent)
    : QOpenGLWidget(parent), m_computeProgram(nullptr), m_mixProgram(nullptr), m_renderProgram(nullptr), m_computeTexture(0), m_imageTexture(0),
      m_cudaTexture(0), m_cudaResource(nullptr), m_screenVBO(QOpenGLBuffer::VertexBuffer), m_width(800), m_height(800) {
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
    if (m_cudaTexture) {
        glDeleteTextures(1, &m_cudaTexture);
        m_cudaTexture = 0;
    }
    delete m_computeProgram;
    delete m_renderProgram;

    if (m_cudaResource) {
        checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaResource), "cudaGraphicsUnregisterResource");
        m_cudaResource = nullptr;
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
    createTexture(&m_cudaTexture, m_width, m_height, 2);    // binding = 2

    // 将纹理注册到 cuda
    checkCudaErrors(cudaGraphicsGLRegisterImage(&m_cudaResource, m_cudaTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore),
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
    if (m_cudaTexture) {
        glDeleteTextures(1, &m_cudaTexture);
        m_cudaTexture = 0;
    }

    // 生成新的纹理
    createTexture(&m_computeTexture, m_width, m_height, 0);
    createTexture(&m_imageTexture, m_width, m_height, 1);
    createTexture(&m_cudaTexture, m_width, m_height, 2);

    if (m_cudaResource) {
        checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaResource), "cudaGraphicsUnregisterResource");
        m_cudaResource = nullptr;
    }
    // 将纹理注册到 cuda
    checkCudaErrors(cudaGraphicsGLRegisterImage(&m_cudaResource, m_cudaTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore),
                    "cudaGraphicsGLRegisterImage");

    // 重置帧计数
    m_frameCount = 0;

    glViewport(0, 0, w, h);
}

void ptScene::paintGL() {
    // -------------------------------
    // 1. 调度计算着色器更新纹理
    // -------------------------------
    // computeShaderPass();
    cudaPass();

    // -------------------------
    // 2. 将新计算结果与之前的混合
    // -------------------------
    mixShaderPass();

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
    // -----------------
    // 编译计算着色器程序
    // -----------------
    m_computeProgram = new QOpenGLShaderProgram();
    if (!m_computeProgram->addShaderFromSourceFile(QOpenGLShader::Compute, "/home/neroued/PathTracingTutorial/Part_2/shaders/pt_compute.glsl")) {
        qWarning() << "Failed to compile compute shader:" << m_computeProgram->log();
        return;
    }
    if (!m_computeProgram->link()) {
        qWarning() << "Failed to link compute shader program:" << m_computeProgram->log();
        return;
    }

    // -----------------
    // 编译混合着色器程序
    // -----------------
    m_mixProgram = new QOpenGLShaderProgram();
    if (!m_mixProgram->addShaderFromSourceFile(QOpenGLShader::Compute, "/home/neroued/PathTracingTutorial/Part_2/shaders/pt_mix.glsl")) {
        qWarning() << "Failed to compile compute shader:" << m_mixProgram->log();
        return;
    }
    if (!m_mixProgram->link()) {
        qWarning() << "Failed to link compute shader program:" << m_mixProgram->log();
        return;
    }

    // ----------------------
    // 编译顶点与片段着色器程序
    // ----------------------
    m_renderProgram = new QOpenGLShaderProgram();
    if (!m_renderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "/home/neroued/PathTracingTutorial/Part_2/shaders/pt_vertex.glsl")) {
        qWarning() << "Failed to compile vertex shader:" << m_renderProgram->log();
        return;
    }
    if (!m_renderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "/home/neroued/PathTracingTutorial/Part_2/shaders/pt_fragment.glsl")) {
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

void ptScene::computeShaderPass() {
    m_computeProgram->bind();

    // 计算工作组数，保证覆盖整个纹理区域（计算着色器中 local_size 为 16×16）
    GLuint groupX = (m_width + 15) / 16;
    GLuint groupY = (m_height + 15) / 16;
    glDispatchCompute(groupX, groupY, 1);

    // 内存屏障，确保计算着色器写入完成后 fragment shader 能正确采样
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    m_computeProgram->release();
}

// pathTracingKernel.cu 提供的接口, 调用 __global__ 函数
extern "C" void launchKernel(cudaSurfaceObject_t surface, int width, int height);

void ptScene::cudaPass() {
    // 将 OpenGL 的纹理资源映射到 CUDA
    checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaResource, 0), "cudaGraphicsMapResources");

    // 获取映射后的 cudaArray
    cudaArray_t textureArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&textureArray, m_cudaResource, 0, 0), "cudaGraphicsSubResourceGetMappedArray");

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
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaResource, 0), "cudaGraphicsUnmapResources");
}

void ptScene::mixShaderPass() {
    m_mixProgram->bind();

    // 传入帧计数
    glUniform1ui(glGetUniformLocation(m_mixProgram->programId(), "sampleCount"), m_frameCount + 1);

    // 计算工作组数，保证覆盖整个纹理区域（计算着色器中 local_size 为 16×16）
    GLuint groupX = (m_width + 15) / 16;
    GLuint groupY = (m_height + 15) / 16;
    glDispatchCompute(groupX, groupY, 1);

    // 内存屏障，确保计算着色器写入完成后 fragment shader 能正确采样
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    m_mixProgram->release();
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