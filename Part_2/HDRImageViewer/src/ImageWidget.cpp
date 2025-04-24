#include "ImageWidget.h"
#include <QDebug>
#include <QApplication>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QtMath>
#include <cstddef>


// 定义用于渲染全屏(或视口)四边形的顶点数据
const float quadVertices[] = {-1.0f, 1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 0.0f,
                              -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  1.0f, 1.0f};

ImageWidget::ImageWidget(QWidget* parent) : QOpenGLWidget(parent), m_vbo(QOpenGLBuffer::VertexBuffer) {
    Q_INIT_RESOURCE(shaders);
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
}

ImageWidget::~ImageWidget() { cleanUp(); }

void ImageWidget::setImageData(const float* data, int width, int height) {
    if (!data || width <= 0 || height <= 0) {
        qWarning() << "setImageData: Invalid image data";
        return;
    }

    // 标记视图矩阵是否需要更新
    // 若图像尺寸不变，则不更新
    if (width != m_imageWidth || height != m_imageHeight) {
        m_scale                 = 1.0f;
        m_translation           = {0.0f, 0.0f};
        m_viewMatrixNeedsUpdate = true;
    }

    m_imageWidth  = width;
    m_imageHeight = height;

    // 复制数据
    size_t dataSize = static_cast<size_t>(width) * height * 3;
    m_imageData.assign(data, data + dataSize);

    // 标记纹理需要更新
    m_textureNeedsUpdate = true;

    // 发送信号
    emit imageInfoChanged(m_imageWidth, m_imageHeight);
    emit scaleChanged(m_scale);
    emit pixelInfoCleared();

    // 请求重绘
    update();
}

void ImageWidget::updateTexture() {
    // 仅在 paintGL 中调用，上下文有保证
    // 首先检查当前的纹理是否有效且匹配
    if (m_texture && m_texture->width() == m_imageWidth && m_texture->height() == m_imageHeight) {
        // 当前纹理可以直接使用
    } else {
        // 需要创建新的纹理
        delete m_texture;
        m_texture = nullptr;

        m_texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
        m_texture->setFormat(QOpenGLTexture::RGB32F); // 三通道 HDR 格式
        m_texture->setSize(m_imageWidth, m_imageHeight);
        m_texture->setWrapMode(QOpenGLTexture::ClampToEdge);
        m_texture->setMinificationFilter(QOpenGLTexture::Nearest);                // 线性插值
        m_texture->setMagnificationFilter(QOpenGLTexture::Nearest);
        m_texture->allocateStorage(QOpenGLTexture::RGB, QOpenGLTexture::Float32); // 分配内存
    }

    // 上传新的数据
    m_texture->bind();
    m_texture->setData(0, QOpenGLTexture::RGB, QOpenGLTexture::Float32, m_imageData.data());
    m_texture->release();

    m_textureNeedsUpdate = false;
}

void ImageWidget::initializeGL() {
    // 初始化 OpenGL 上下文
    if (!initializeOpenGLFunctions()) {
        qCritical() << "Failed to initialize OpenGL functions.";
        QApplication::quit();
        return;
    }

    // 设置清屏颜色（深灰色）
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // 加载着色器
    if (!loadShaders()) { qCritical() << "Failed to load shaders."; }

    // 初始化 OpenGL 资源
    initializeOpenGLResources();
}

void ImageWidget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
    m_viewMatrixNeedsUpdate = true;
}

void ImageWidget::paintGL() {
    // 若传入了新的图像，处理新图像
    if (m_textureNeedsUpdate) { updateTexture(); }

    // 处理视图矩阵的变化
    if (m_viewMatrixNeedsUpdate) { updateViewMatrix(); }

    // 清屏
    glClear(GL_COLOR_BUFFER_BIT);

    // 检查着色器与纹理状态
    if (!m_shaderProgram || !m_shaderProgram->isLinked() || !m_texture || !m_texture->isCreated()) { return; }

    // 绘制
    m_shaderProgram->bind();
    glActiveTexture(GL_TEXTURE0);
    m_texture->bind();

    // 设置 uniforms
    m_shaderProgram->setUniformValue("hdrTexture", 0);
    m_shaderProgram->setUniformValue("exposure", m_exposure);
    m_shaderProgram->setUniformValue("viewMatrix", m_viewMatrix);

    // 绘制四边形
    m_vao.bind();
    glDrawArrays(GL_TRIANGLES, 0, 6);

    m_texture->release();
    m_shaderProgram->release();
    m_vao.release();
}

void ImageWidget::cleanUp() {
    makeCurrent();
    delete m_shaderProgram;
    delete m_texture;
    m_shaderProgram = nullptr;
    m_texture       = nullptr;
    m_vbo.destroy();
    m_vao.destroy();
    doneCurrent();
}

void ImageWidget::updateViewMatrix() {
    // --- 计算宽高比 ---
    float widgetWidth  = static_cast<float>(width());  // 获取当前 widget 宽度
    float widgetHeight = static_cast<float>(height()); // 获取当前 widget 高度

    // 防止除零
    if (m_imageWidth <= 0 || m_imageHeight <= 0 || widgetWidth <= 0 || widgetHeight <= 0) {
        m_viewMatrix.setToIdentity(); // 无效尺寸时设为单位矩阵
        m_viewMatrixNeedsUpdate = false;
        return;
    }

    float imageAspect  = static_cast<float>(m_imageWidth) / m_imageHeight;
    float widgetAspect = widgetWidth / widgetHeight;

    float scaleX = m_scale;
    float scaleY = m_scale;

    // --- 计算校正因子 ---
    if (widgetAspect > imageAspect) {
        // Widget 比图像更宽 (或者说图像相对更高)
        // 图像高度填满 widget 高度，宽度需要根据图像比例缩减
        // (widgetAspect / imageAspect) > 1
        scaleX = m_scale * (imageAspect / widgetAspect); // 缩小 X 方向的比例
        scaleY = m_scale;
    } else {
        // Widget 比图像更高 (或者说图像相对更宽)
        // 图像宽度填满 widget 宽度，高度需要根据图像比例缩减
        // (imageAspect / widgetAspect) > 1
        scaleX = m_scale;
        scaleY = m_scale * (widgetAspect / imageAspect); // 缩小 Y 方向的比例
    }

    // --- 应用变换 ---
    m_viewMatrix.setToIdentity();
    // 1. 应用平移 (注意：平移是在校正后的坐标系中进行的)
    m_viewMatrix.translate(m_translation.x(), m_translation.y(), 0.0f);
    // 2. 应用考虑了宽高比的非均匀缩放
    m_viewMatrix.scale(scaleX, scaleY, 1.0f);
    // 计算逆矩阵
    m_invViewMatrix = m_viewMatrix.inverted();

    m_viewMatrixNeedsUpdate = false;
}

void ImageWidget::initializeOpenGLResources() {
    makeCurrent();

    // 创建 vao
    m_vao.create();
    m_vao.bind();

    // 创建并分配 vbo
    m_vbo.create();
    m_vbo.bind();
    m_vbo.allocate(quadVertices, sizeof(quadVertices));

    // 设置顶点属性
    // 属性 0：位置（2 个浮点数），偏移 0，步长为 4*sizeof(float)
    m_shaderProgram->enableAttributeArray(0);
    m_shaderProgram->setAttributeArray(0, GL_FLOAT, 0, 2, 4 * sizeof(float));
    // 属性 1：纹理坐标（2 个浮点数），偏移为 2*sizeof(float)
    m_shaderProgram->enableAttributeArray(1);
    m_shaderProgram->setAttributeBuffer(1, GL_FLOAT, 2 * sizeof(float), 2, 4 * sizeof(float));

    m_vbo.release();
    m_vao.release();

    doneCurrent();
}

bool ImageWidget::loadShaders() {
    // 必须在正确的 OpenGL 上下文中操作
    makeCurrent();

    m_shaderProgram = new QOpenGLShaderProgram(this);

    // 从文件加载顶点着色器
    if (!m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/vertex.glsl")) {
        qCritical() << "Failed to compile vertex shader:" << m_shaderProgram->log();
        delete m_shaderProgram;
        m_shaderProgram = nullptr;
        doneCurrent();
        return false;
    }

    // 从文件加载片段着色器
    if (!m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/fragment.glsl")) {
        qCritical() << "Failed to compile fragment shader:" << m_shaderProgram->log();
        delete m_shaderProgram;
        m_shaderProgram = nullptr;
        doneCurrent();
        return false;
    }

    // 链接着色器程序
    if (!m_shaderProgram->link()) {
        qCritical() << "Failed to link shader program:" << m_shaderProgram->log();
        delete m_shaderProgram;
        m_shaderProgram = nullptr;
        doneCurrent();
        return false;
    }

    doneCurrent();

    return true;
}

void ImageWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        m_panning      = true;
        m_lastMousePos = event->position();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
    } else {
        QWidget::mousePressEvent(event);
    }
}

void ImageWidget::mouseMoveEvent(QMouseEvent* event) {
    QPointF currentPos = event->position();

    // 处理图像的位移
    if (m_panning) {
        QPointF delta = currentPos - m_lastMousePos;

        // 将鼠标的移动的像素转换为 NDC 的位移
        float ndcDeltaX = (delta.x() / width()) * 2.0f;
        float ndcDeltaY = (-delta.y() / height()) * 2.0f; // y 需要翻转

        m_translation += QVector2D(ndcDeltaX, ndcDeltaY);

        m_lastMousePos          = currentPos;
        m_viewMatrixNeedsUpdate = true;
        update();
        event->accept();
    } else {
        QWidget::mouseMoveEvent(event);
    }

    // 处理像素信息的变化
    if (m_imageWidth > 0 && m_imageHeight > 0) {
        QPointF imageCoords = mapWidgetToImageCoords(currentPos);
        int imgX            = qFloor(imageCoords.x());
        int imgY            = qFloor(imageCoords.y());

        float r, g, b;
        if (getPixelColorAt(imgX, imgY, r, g, b)) {
            emit pixelInfoUpdated(imgX, imgY, r, g, b);
        } else {
            emit pixelInfoCleared();
        }
    } else {
        emit pixelInfoCleared();
    }
}

void ImageWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && m_panning) {
        m_panning = false;
        setCursor(Qt::ArrowCursor);
        event->accept();
    } else {
        QWidget::mouseReleaseEvent(event);
    }
}

void ImageWidget::wheelEvent(QWheelEvent* event) {
    int delta = event->angleDelta().y();

    if (delta != 0) {
        float scaleFactor = qPow(1.0008, delta);
        float newScale    = m_scale * scaleFactor;

        // 计算鼠标位置的 NDC 坐标，实现以鼠标中心缩放
        QPointF mousePos = event->position();
        float ndcX       = (mousePos.x() / width()) * 2.0f - 1.0f;
        float ndcY       = 1.0f - (mousePos.y() / height()) * 2.0f;
        QVector2D mouseNDC(ndcX, ndcY);

        m_translation = mouseNDC * (1.0f - scaleFactor) + m_translation * scaleFactor;

        m_scale                 = newScale;
        m_viewMatrixNeedsUpdate = true;

        update();
        event->accept();

        emit scaleChanged(m_scale);
    } else {
        QWidget::wheelEvent(event);
    }
}

void ImageWidget::leaveEvent(QEvent* event) {
    QWidget::leaveEvent(event);
    emit pixelInfoCleared();
}

bool ImageWidget::getPixelColorAt(int x, int y, float& r, float& g, float& b) const {
    if (m_imageData.empty() || m_imageWidth <= 0 || m_imageHeight <= 0) { return false; }

    if (x < 0 || x >= m_imageWidth || y < 0 || y >= m_imageHeight) { return false; }

    size_t index = (static_cast<size_t>(y) * m_imageWidth + static_cast<size_t>(x)) * 3;

    r = m_imageData[index + 0];
    g = m_imageData[index + 1];
    b = m_imageData[index + 2];

    return true;
}

QPointF ImageWidget::mapWidgetToImageCoords(const QPointF& widgetPos) const {
    if (m_imageWidth <= 0 || m_imageHeight <= 0 || width() <= 0 || height() <= 0) {
        return {-1.0f, -1.0f};
    }

    // 计算 NDC 坐标
    float ndcX = (widgetPos.x() / width()) * 2.0f - 1.0f;
    float ndcY = 1.0f - (widgetPos.y() / height()) * 2.0f;
    
    // 将 NDC 坐标转换为模型坐标
    QVector3D modelCoords = m_invViewMatrix.map(QVector3D(ndcX, ndcY, 0.0f));

    // 将模型坐标转换为图像像素坐标
    float texU = (modelCoords.x() + 1.0f) * 0.5f;
    float texV = (modelCoords.y() + 1.0f) * 0.5f;
    float imgX = texU * m_imageWidth;
    float imgY = (1.0f - texV) * m_imageHeight;

    return {imgX, imgY};
}

void ImageWidget::setExposure(float exposure) {
    m_exposure = exposure;
    update();
    emit exposureChanged(m_exposure);
}

void ImageWidget::resetViewAndParams() {
    m_scale = 1.0f;
    emit scaleChanged(m_scale);

    m_exposure = 1.0f;
    emit exposureChanged(m_exposure);

    m_translation = {0.0f, 0.0f};
    m_viewMatrixNeedsUpdate = true;
    update();
    emit pixelInfoCleared();
}