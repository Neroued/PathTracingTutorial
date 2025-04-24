cmake_minimum_required(VERSION 3.16)
project(HDRImageViewer LANGUAGES CXX)

# ========== 选择正确的 Qt ==========
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{QT_MSVC}")
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)
find_package(OpenGL REQUIRED)

# ========== 启用 Qt AUTOGEN ==========
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ========== 1) 定义 OBJECT 库，触发自身编译（含 MOC/RCC） ==========
add_library(HDRImageViewer_objs OBJECT
    src/ControlPanel.cpp
    src/HDRImageViewer.cpp
    src/ImageWidget.cpp

    include/ControlPanel.h # 含 Q_OBJECT
    include/HDRImageViewer.h # 含 Q_OBJECT
    include/ImageWidget.h # 含 Q_OBJECT

    shaders/shaders.qrc # Qt 资源
)

# 子模块内部编译时需要的 include 路径
target_include_directories(HDRImageViewer_objs
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(HDRImageViewer_objs
    PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    OpenGL::GL
)

# C++17
target_compile_features(HDRImageViewer_objs
    PRIVATE
    cxx_std_17
)

# ========== 2) 定义 INTERFACE 包装库，暴露给上层 ==========
add_library(HDRImageViewer INTERFACE)

# 把 OBJECT 库编译出的所有 object 加入 INTERFACE 源
target_sources(HDRImageViewer
    INTERFACE
    $<TARGET_OBJECTS:HDRImageViewer_objs>
)

# 暴露头文件给上层使用（BUILD_INTERFACE 确保 install 后不会误用源目录）
target_include_directories(HDRImageViewer
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# 把 Qt 和 OpenGL 的使用需求也暴露给上层
target_link_libraries(HDRImageViewer
    INTERFACE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    OpenGL::GL
)

# C++17 也通过 INTERFACE 传播
target_compile_features(HDRImageViewer
    INTERFACE
    cxx_std_17
)
