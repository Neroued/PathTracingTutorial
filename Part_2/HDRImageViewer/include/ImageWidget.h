#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_5_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <vector>
#include <QVector2D>
#include <QMatrix4x4>

class ImageWidget : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core {
    Q_OBJECT

public:
    explicit ImageWidget(QWidget* parent = nullptr);
    ~ImageWidget() override;

    // 设置图像数据的接口
    // data: float* 指向 HDR 图像数据的指针，包含三个通道
    void setImageData(const float* data, int width, int height);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    // 鼠标事件
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void leaveEvent(QEvent* event) override;

public:
signals:
    // 图像尺寸改变
    void imageInfoChanged(int width, int height);
    // 缩放改变
    void scaleChanged(float scale);
    // 当鼠标在图像上移动，发送坐标和 RGB 信息
    void pixelInfoUpdated(int imgX, int imgY, float r, float g, float b);
    // 当鼠标离开图像区域或不存在有效图像
    void pixelInfoCleared();
    // 曝光改变
    void exposureChanged(float exposure);

public slots:
    void setExposure(float exposure);
    void resetViewAndParams();

private:
    // 初始化 OpenGL 资源
    void initializeOpenGLResources();
    // 编译着色器
    bool loadShaders();
    // 清理资源
    void cleanUp();
    // 更新 m_texture
    void updateTexture();
    // 更新视图矩阵
    void updateViewMatrix();
    // 获取在屏幕 (x, y) 处的像素信息
    bool getPixelColorAt(int x, int y, float& r, float& g, float& b) const;
    // 将屏幕坐标转换为图像坐标
    QPointF mapWidgetToImageCoords(const QPointF& widgetPos) const;

private:
    // --- OpenGL 相关 ---//
    QOpenGLShaderProgram* m_shaderProgram = nullptr;
    QOpenGLTexture* m_texture             = nullptr;
    QOpenGLVertexArrayObject m_vao;
    QOpenGLBuffer m_vbo;

    // --- 图像数据 --- //
    std::vector<float> m_imageData;
    int m_imageWidth             = 0;
    int m_imageHeight            = 0;
    bool m_textureNeedsUpdate    = false;
    bool m_viewMatrixNeedsUpdate = true;

    // --- 交互相关 --- //
    float m_scale           = 1.0f;
    QVector2D m_translation = {0.0f, 0.0f};
    QMatrix4x4 m_viewMatrix;
    QMatrix4x4 m_invViewMatrix;
    QPointF m_lastMousePos;
    bool m_panning = false;

    // 渲染参数
    float m_exposure = 1.0f;
};