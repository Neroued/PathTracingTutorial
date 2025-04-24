// MainWindow.cpp
#include "config.h"
#include "MainWindow.h"
#include "Scene.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSlider>
#include <QLabel>
#include <QGroupBox>
#include <QPushButton>

BEGIN_NAMESPACE_PT

MainWindow::MainWindow(QWidget* parent) : QWidget(parent) {
    // 主布局为水平布局，设置适当的间距和边距
    QHBoxLayout* mainLayout = new QHBoxLayout(this);
    mainLayout->setSpacing(5);
    mainLayout->setContentsMargins(5, 5, 5, 5);

    // 左侧控制面板
    QGroupBox* controlPanel = new QGroupBox("Shader 控制", this);
    controlPanel->setMinimumWidth(250);
    QVBoxLayout* controlLayout = new QVBoxLayout(controlPanel);
    controlLayout->setSpacing(5);
    controlLayout->setContentsMargins(5, 5, 5, 5);
    controlLayout->setAlignment(Qt::AlignTop);

    // Exposure 控制 —— 标签与数值放在同一行
    QLabel* exposureLabel = new QLabel("Exposure", controlPanel);
    // 映射关系：exposure = value / 10.0f - 7.8f
    QLabel* exposureValueLabel        = new QLabel(QString::number(static_cast<float>(50) / 10.0f - 7.8f), controlPanel);
    QHBoxLayout* exposureHeaderLayout = new QHBoxLayout();
    exposureHeaderLayout->setSpacing(5);
    exposureHeaderLayout->addWidget(exposureLabel);
    exposureHeaderLayout->addWidget(exposureValueLabel);
    controlLayout->addLayout(exposureHeaderLayout);

    // Exposure 滑块放在下一行
    QSlider* exposureSlider = new QSlider(Qt::Horizontal, controlPanel);
    exposureSlider->setRange(0, 1000);
    exposureSlider->setValue(500);
    controlLayout->addWidget(exposureSlider);

    // Gamma 控制 —— 标签与数值放在同一行
    QLabel* gammaLabel = new QLabel("Gamma", controlPanel);
    // 映射关系：gamma = value / 50.0f + 1.2f
    QLabel* gammaValueLabel        = new QLabel(QString::number(static_cast<float>(50) / 50.0f + 1.2f), controlPanel);
    QHBoxLayout* gammaHeaderLayout = new QHBoxLayout();
    gammaHeaderLayout->setSpacing(5);
    gammaHeaderLayout->addWidget(gammaLabel);
    gammaHeaderLayout->addWidget(gammaValueLabel);
    controlLayout->addLayout(gammaHeaderLayout);

    // Gamma 滑块放在下一行
    QSlider* gammaSlider = new QSlider(Qt::Horizontal, controlPanel);
    gammaSlider->setRange(0, 1000);
    gammaSlider->setValue(500);
    controlLayout->addWidget(gammaSlider);

    // 添加重置按钮，将两个滑块的值重置为50
    QPushButton* resetButton = new QPushButton("重置", controlPanel);
    controlLayout->addWidget(resetButton);

    // 将控制面板加入主布局
    mainLayout->addWidget(controlPanel);

    // 右侧添加 Scene，并固定其尺寸
    m_scene = new Scene(this);
    m_scene->setFixedSize(m_scene->size());
    mainLayout->addWidget(m_scene, 1); // 伸缩因子设为 1

    // 信号与槽连接：更新 shader 参数，并更新滑块对应的数值显示（显示映射后的值）
    connect(exposureSlider, &QSlider::valueChanged, this, [=](int value) {
        float exposure = static_cast<float>(value) / 100.0f - 7.8f;
        exposureValueLabel->setText(QString::number(exposure));
        m_scene->setExposure(exposure);
    });

    connect(gammaSlider, &QSlider::valueChanged, this, [=](int value) {
        float gamma = static_cast<float>(value) / 1000.0f * 4.4f;
        gammaValueLabel->setText(QString::number(gamma));
        m_scene->setGamma(gamma);
    });

    // 重置按钮点击信号：将两个滑块的值设置为 500
    connect(resetButton, &QPushButton::clicked, this, [=]() {
        exposureSlider->setValue(500);
        gammaSlider->setValue(500);
    });

    setLayout(mainLayout);
}

END_NAMESPACE_PT
