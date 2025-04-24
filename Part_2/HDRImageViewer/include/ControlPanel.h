#pragma once

#include <QWidget>
#include <QLabel>
#include <QSlider>
#include <QPushButton>

class ControlPanel : public QWidget {
    Q_OBJECT

public:
    explicit ControlPanel(QWidget* parent = nullptr);
    ~ControlPanel() override = default;

public:
signals:
    // 通过滑块修改曝光发送信号
    void exposureChangeRequested(float exposure);
    // 重置
    void resetRequested();

public slots:
    // 更新图像信息
    void updateImageInfo(int width, int height);

    // 更新缩放
    void updateScale(float scale);

    // 更新像素位置与颜色信息
    void updatePixelInfo(int imgX, int imgY, float r, float g, float b);

    // 清除显示的像素信息
    void clearPixelInfo();

    // 更新曝光滑块的数值 
    void updateExposureDisplay(float exposure);

private slots:
    // 内部连接曝光滑块
    void onExposureSliderChanged(int value);

    // 内部连接重置按钮
    void onResetButtonClicked();

private:
    void setupUi();

    float sliderValueToExposure(int value) const;
    int exposureToSliderValue(float exposure) const;

private:
    QLabel* m_imageSizeLabel;
    QLabel* m_scaleLabel;
    QLabel* m_pixelPosLabel;
    QLabel* m_pixelColorLabel;

    QSlider* m_exposureSlider;
    QLabel* m_exposureLabel;

    QPushButton* m_resetButton;

    const int EXPOSURE_SLIDER_MIN      = 0;
    const int EXPOSURE_SLIDER_MAX      = 2000;
    const float EXPOSURE_MIN_VALUE     = 0.0f;
    const float EXPOSURE_MAX_VALUE     = 5.0f;
    const float EXPOSURE_DEFAULT_VALUE = 1.0f;
};