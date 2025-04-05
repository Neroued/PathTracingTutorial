#pragma once

#include "config.h"
#include <QWidget>
#include "Scene.h"

class QSlider;
class QLabel;

BEGIN_NAMESPACE_PT

class MainWindow : public QWidget {
    Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr);
    Scene* m_scene;
};

END_NAMESPACE_PT