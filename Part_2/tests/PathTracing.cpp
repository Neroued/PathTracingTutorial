#include <MainWindow.h>
#include <QApplication>
#include <QSurfaceFormat>

#include <QGuiApplication>

int main(int argc, char* argv[]) {
    // 禁用 Qt 的高 DPI 自动缩放（双保险：环境变量 + 属性）
    qputenv("QT_ENABLE_HIGHDPI_SCALING", "0");
    qputenv("QT_AUTO_SCREEN_SCALE_FACTOR", "0");
    qputenv("QT_SCALE_FACTOR", "1");
    QSurfaceFormat format;
    format.setVersion(4, 5);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setSwapInterval(0);
    QSurfaceFormat::setDefaultFormat(format);

    // 禁用 Qt 自身的高 DPI 缩放处理
    QCoreApplication::setAttribute(Qt::AA_DisableHighDpiScaling);
    // 避免缩放因子取整导致异常放大
    QGuiApplication::setHighDpiScaleFactorRoundingPolicy(
        Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);

    QApplication app(argc, argv);

    pt::MainWindow mainWindow;
    mainWindow.m_scene->loadScene();

    mainWindow.show();
    return app.exec();
}