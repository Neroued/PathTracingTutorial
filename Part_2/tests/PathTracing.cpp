#include <MainWindow.h>
#include <QApplication>
#include <QSurfaceFormat>

int main(int argc, char* argv[]) {
    QSurfaceFormat format;
    format.setVersion(4, 5);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setSwapInterval(0);
    QSurfaceFormat::setDefaultFormat(format);

    QApplication app(argc, argv);

    pt::MainWindow mainWindow;
    mainWindow.m_scene->loadScene();

    mainWindow.show();
    return app.exec();
}