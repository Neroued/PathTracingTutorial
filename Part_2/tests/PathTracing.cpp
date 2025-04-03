#include <Scene.h>
#include <QApplication>
#include <QSurfaceFormat>

int main(int argc, char* argv[]) {
    QSurfaceFormat format;
    format.setVersion(4, 5);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setSwapInterval(0);
    QSurfaceFormat::setDefaultFormat(format);

    QApplication app(argc, argv);

    pt::Scene pt;
    pt.setFixedSize(pt.size());

    pt.loadScene();

    pt.show();
    return app.exec();
}