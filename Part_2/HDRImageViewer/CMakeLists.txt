cmake_minimum_required(VERSION 3.16)
project(HDRImageViewer LANGUAGES CXX)

# ========== 选择正确的 Qt ==========
if(DEFINED ENV{QT_MSVC})
    set(QT_ROOT "$ENV{QT_MSVC}")
elseif(DEFINED ENV{QT_MINGW})
    set(QT_ROOT "$ENV{QT_MINGW}")
endif()
if(QT_ROOT)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${QT_ROOT}")
endif()
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)
find_package(OpenGL REQUIRED)

# ========== 启用 Qt AUTOGEN ==========
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ========== 定义一个常规 STATIC 库，内部完成 MOC/RCC ==========
add_library(HDRImageViewer STATIC
    src/ControlPanel.cpp
    src/HDRImageViewer.cpp
    src/ImageWidget.cpp

    include/ControlPanel.h
    include/HDRImageViewer.h
    include/ImageWidget.h

    shaders/shaders.qrc
)

target_include_directories(HDRImageViewer
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(HDRImageViewer
    PUBLIC
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    OpenGL::GL
)

target_compile_features(HDRImageViewer
    PUBLIC
    cxx_std_17
)
