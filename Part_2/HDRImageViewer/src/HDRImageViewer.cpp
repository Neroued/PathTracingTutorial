#include "HDRImageViewer.h"

HDRImageViewer::HDRImageViewer(QWidget* parent) : QWidget(parent) {
    setupUi();
    setupConnections();
    resize(1000, 800);
}

void HDRImageViewer::setupUi() {
    m_imageWidget  = new ImageWidget(this);
    m_controlPanel = new ControlPanel(this);

    m_mainLayout = new QHBoxLayout(this);

    m_mainLayout->addWidget(m_controlPanel, 0);
    m_mainLayout->addWidget(m_imageWidget, 1);

    m_mainLayout->setContentsMargins(5, 5, 5, 5);
    m_mainLayout->setSpacing(5);
}

void HDRImageViewer::setImageData(const float* data, int width, int height) { m_imageWidget->setImageData(data, width, height); }

void HDRImageViewer::setupConnections() {
    // ImageWidget -> ControlPanel (Information Flow)
    connect(m_imageWidget, &ImageWidget::imageInfoChanged, m_controlPanel, &ControlPanel::updateImageInfo);
    connect(m_imageWidget, &ImageWidget::scaleChanged, m_controlPanel, &ControlPanel::updateScale);
    connect(m_imageWidget, &ImageWidget::pixelInfoUpdated, m_controlPanel, &ControlPanel::updatePixelInfo);
    connect(m_imageWidget, &ImageWidget::pixelInfoCleared, m_controlPanel, &ControlPanel::clearPixelInfo);
    connect(m_imageWidget, &ImageWidget::exposureChanged, m_controlPanel, &ControlPanel::updateExposureDisplay);

    // ControlPanel -> ImageWidget (Control Flow)
    connect(m_controlPanel, &ControlPanel::exposureChangeRequested, m_imageWidget, &ImageWidget::setExposure);
    connect(m_controlPanel, &ControlPanel::resetRequested, m_imageWidget, &ImageWidget::resetViewAndParams);
}