#include <ptScene.h>
#include <QDebug>
#include <qflags.h>

ptScene::ptScene(QWidget* parent)
    : QOpenGLWidget(parent), m_computeProgram(nullptr), m_renderProgram(nullptr), m_computeTexture(0), m_screenVBO(QOpenGLBuffer::VertexBuffer),
      m_width(800), m_height(800) {
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
    delete m_computeProgram;
    delete m_renderProgram;

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

    // 生成新的纹理
    createTexture(&m_computeTexture, m_width, m_height, 0);

    glViewport(0, 0, w, h);
}

void ptScene::paintGL() {
    // -------------------------------
    // 1. 调度计算着色器更新纹理
    // -------------------------------
    computeShaderPass();

    // -------------------------------
    // 2. 渲染全屏四边形，显示计算结果纹理
    // -------------------------------
    renderShaderPass();

    // -----------------------
    // 3. 调用 update() 触发重绘
    // -----------------------
    update();
}

void ptScene::compileShaders() {
    // -----------------
    // 编译计算着色器程序
    // -----------------
    m_computeProgram = new QOpenGLShaderProgram();
    if (!m_computeProgram->addShaderFromSourceFile(QOpenGLShader::Compute, "/home/neroued/PathTracingTutorial/Part_0/shaders/pt_compute.glsl")) {
        qFatal() << "Failed to compile compute shader:" << m_computeProgram->log();
        return;
    }
    if (!m_computeProgram->link()) {
        qFatal() << "Failed to link compute shader program:" << m_computeProgram->log();
        return;
    }

    // ----------------------
    // 编译顶点与片段着色器程序
    // ----------------------
    m_renderProgram = new QOpenGLShaderProgram();
    if (!m_renderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "/home/neroued/PathTracingTutorial/Part_0/shaders/pt_vertex.glsl")) {
        qFatal() << "Failed to compile vertex shader:" << m_renderProgram->log();
        return;
    }
    if (!m_renderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "/home/neroued/PathTracingTutorial/Part_0/"
                                                                           "shaders/pt_fragment.glsl")) {
        qFatal() << "Failed to compile fragment shader:" << m_renderProgram->log();
        return;
    }
    if (!m_renderProgram->link()) {
        qFatal() << "Failed to link render shader program:" << m_renderProgram->log();
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

void ptScene::renderShaderPass() {
    // 使用着色器绘制全屏四边形并采样compute shader计算的texture
    m_renderProgram->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    m_screenVAO.bind();

    // 激活并绑定纹理到纹理单元 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_computeTexture);
    // 将 uniform "renderedTexture" 绑定到纹理单元 0
    m_renderProgram->setUniformValue("renderedTexture", 0);

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