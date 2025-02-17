#include <QApplication>
#include <QSurfaceFormat>
#include <ptScene.h>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QSurfaceFormat format;
    format.setVersion(4, 5);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);

    ptScene pt;
    pt.resize(800, 800);

    pt.show();
    return app.exec();
}