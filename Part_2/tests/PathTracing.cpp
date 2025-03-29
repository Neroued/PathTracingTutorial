#include <Scene.h>
#include <QApplication>
#include <QSurfaceFormat>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    QSurfaceFormat format;
    format.setVersion(4, 5);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);

    pt::Scene pt;
    pt.resize(800, 800);

    pt.show();
    return app.exec();
}