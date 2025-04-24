#include "ControlPanel.h"
#include <QLocale>
#include <QBoxLayout>
#include <QGroupBox>
#include <algorithm>
#include <cmath>

ControlPanel::ControlPanel(QWidget* parent) : QWidget(parent) {
    setupUi();

    updateImageInfo(0, 0);
    updateScale(1.0f);
    clearPixelInfo();
    updateExposureDisplay(EXPOSURE_DEFAULT_VALUE);
}

void ControlPanel::setupUi() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(10);

    QLocale locale(QLocale::English, QLocale::UnitedStates);

    // 图像信息 Group
    QGroupBox* infoGroup    = new QGroupBox("Image Information");
    QVBoxLayout* infoLayout = new QVBoxLayout(infoGroup);

    QHBoxLayout* sizeLayout = new QHBoxLayout();
    sizeLayout->addWidget(new QLabel("Dimensions:"));
    m_imageSizeLabel = new QLabel("N/A");
    m_imageSizeLabel->setAlignment(Qt::AlignRight);
    sizeLayout->addWidget(m_imageSizeLabel);
    infoLayout->addLayout(sizeLayout);

    QHBoxLayout* scaleLayout = new QHBoxLayout();
    scaleLayout->addWidget(new QLabel("Scale Factor:"));
    m_scaleLabel = new QLabel("N/A");
    m_scaleLabel->setAlignment(Qt::AlignRight);
    scaleLayout->addWidget(m_scaleLabel);
    infoLayout->addLayout(scaleLayout);

    mainLayout->addWidget(infoGroup);

    // --- Pixel Info Group ---
    QGroupBox* pixelGroup    = new QGroupBox("Pixel Inspector");
    QVBoxLayout* pixelLayout = new QVBoxLayout(pixelGroup);

    QHBoxLayout* posLayout = new QHBoxLayout();
    posLayout->addWidget(new QLabel("Image Pos (X, Y):"));
    m_pixelPosLabel = new QLabel("N/A");
    m_pixelPosLabel->setAlignment(Qt::AlignRight);
    posLayout->addWidget(m_pixelPosLabel);
    pixelLayout->addLayout(posLayout);

    QHBoxLayout* colorLayout = new QHBoxLayout();
    colorLayout->addWidget(new QLabel("Pixel Color (R, G, B):"));
    m_pixelColorLabel = new QLabel("N/A");
    m_pixelColorLabel->setAlignment(Qt::AlignRight);
    m_pixelColorLabel->setMinimumWidth(150); // Give it some space
    colorLayout->addWidget(m_pixelColorLabel);
    pixelLayout->addLayout(colorLayout);

    mainLayout->addWidget(pixelGroup);


    // --- Controls Group ---
    QGroupBox* controlsGroup    = new QGroupBox("Controls");
    QVBoxLayout* controlsLayout = new QVBoxLayout(controlsGroup);

    // Exposure
    QHBoxLayout* exposureLayout = new QHBoxLayout();
    exposureLayout->addWidget(new QLabel("Exposure:"));
    m_exposureSlider = new QSlider(Qt::Horizontal);
    m_exposureSlider->setRange(EXPOSURE_SLIDER_MIN, EXPOSURE_SLIDER_MAX);
    m_exposureSlider->setValue(exposureToSliderValue(EXPOSURE_DEFAULT_VALUE)); // Set initial position
    m_exposureSlider->setToolTip("Adjust image exposure");
    exposureLayout->addWidget(m_exposureSlider, 1);                            // Slider takes more space

    m_exposureLabel = new QLabel(locale.toString(EXPOSURE_DEFAULT_VALUE, 'f', 2));
    m_exposureLabel->setMinimumWidth(40); // Ensure space for number
    m_exposureLabel->setAlignment(Qt::AlignRight);
    exposureLayout->addWidget(m_exposureLabel);
    controlsLayout->addLayout(exposureLayout);

    // Reset Button
    m_resetButton = new QPushButton("Reset View");
    m_resetButton->setToolTip("Reset zoom, pan, and exposure");
    controlsLayout->addWidget(m_resetButton, 0, Qt::AlignCenter);

    mainLayout->addWidget(controlsGroup);

    mainLayout->addStretch(1);

    // --- Connections ---
    connect(m_exposureSlider, &QSlider::valueChanged, this, &ControlPanel::onExposureSliderChanged);
    connect(m_resetButton, &QPushButton::clicked, this, &ControlPanel::onResetButtonClicked);
}

// --- Slots Implementation ---

void ControlPanel::updateImageInfo(int width, int height) {
    if (width > 0 && height > 0) {
        m_imageSizeLabel->setText(QString("%1 x %2").arg(width).arg(height));
    } else {
        m_imageSizeLabel->setText("N/A");
    }
}

void ControlPanel::updateScale(float scale) {
    QLocale locale(QLocale::English, QLocale::UnitedStates);
    m_scaleLabel->setText(locale.toString(scale, 'f', 2) + "x");
}

void ControlPanel::updatePixelInfo(int imgX, int imgY, float r, float g, float b) {
    QLocale locale(QLocale::English, QLocale::UnitedStates);
    m_pixelPosLabel->setText(QString("(%1, %2)").arg(imgX).arg(imgY));
    // Format float colors - adjust precision as needed
    m_pixelColorLabel->setText(
        QString("(%1, %2, %3)").arg(locale.toString(r, 'f', 3)).arg(locale.toString(g, 'f', 3)).arg(locale.toString(b, 'f', 3)));
}

void ControlPanel::clearPixelInfo() {
    m_pixelPosLabel->setText("N/A");
    m_pixelColorLabel->setText("N/A");
}

void ControlPanel::updateExposureDisplay(float exposure) {
    QLocale locale(QLocale::English, QLocale::UnitedStates);
    m_exposureLabel->setText(locale.toString(exposure, 'f', 2));
    // Update slider position without emitting the signal again
    bool oldSignalState = m_exposureSlider->blockSignals(true);
    m_exposureSlider->setValue(exposureToSliderValue(exposure));
    m_exposureSlider->blockSignals(oldSignalState);
}

void ControlPanel::onExposureSliderChanged(int value) {
    float exposure = sliderValueToExposure(value);
    QLocale locale(QLocale::English, QLocale::UnitedStates);
    m_exposureLabel->setText(locale.toString(exposure, 'f', 2));
    emit exposureChangeRequested(exposure); // Emit signal for ImageWidget
}

void ControlPanel::onResetButtonClicked() {
    emit resetRequested(); // Emit signal for ImageWidget
}

// --- Helper function implementations ---

// Linear mapping example
float ControlPanel::sliderValueToExposure(int value) const {
    float fraction = static_cast<float>(value - EXPOSURE_SLIDER_MIN) / (EXPOSURE_SLIDER_MAX - EXPOSURE_SLIDER_MIN);
    return EXPOSURE_MIN_VALUE + fraction * (EXPOSURE_MAX_VALUE - EXPOSURE_MIN_VALUE);
}

int ControlPanel::exposureToSliderValue(float exposure) const {
    // Clamp exposure to the valid range before converting
    exposure       = std::max(EXPOSURE_MIN_VALUE, std::min(exposure, EXPOSURE_MAX_VALUE));
    float fraction = (exposure - EXPOSURE_MIN_VALUE) / (EXPOSURE_MAX_VALUE - EXPOSURE_MIN_VALUE);
    return EXPOSURE_SLIDER_MIN + static_cast<int>(std::round(fraction * (EXPOSURE_SLIDER_MAX - EXPOSURE_SLIDER_MIN)));
}