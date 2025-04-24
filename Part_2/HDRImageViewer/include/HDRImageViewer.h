#pragma once

#include <QWidget>
#include <QBoxLayout>

#include "ImageWidget.h"
#include "ControlPanel.h"

class HDRImageViewer : public QWidget {
    Q_OBJECT

public:
    explicit HDRImageViewer(QWidget* parent = nullptr);
    ~HDRImageViewer() override = default;

    void setImageData(const float* data, int width, int height);

private:
    void setupUi();

    void setupConnections();

private:
    ImageWidget* m_imageWidget;
    ControlPanel* m_controlPanel;
    QHBoxLayout* m_mainLayout;
};
