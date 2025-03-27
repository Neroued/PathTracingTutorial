cmake_minimum_required(VERSION 3.16)
project(PathTracingTutorial VERSION 1.0.0)


# ========== 设置c++标准 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ========== 查找需要的库 ==========
find_package(OpenGL REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)


# ========== 添加include路径 ==========
include_directories(include/)


# ========== Qt初始化 ==========
qt_standard_project_setup()


# ========== 源文件与头文件 ==========
file(GLOB SOURCES src/*.cpp)
file(GLOB HEADERS include/*.h)


# ========== 添加可执行目标 ==========
qt_add_executable(PathTracing ${HEADERS} ${SOURCES} tests/PathTracing.cpp)


# ========== 链接库 ==========
target_link_libraries(PathTracing PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    OpenGL::GL
)