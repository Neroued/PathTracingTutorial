cmake_minimum_required(VERSION 3.16)
project(PathTracingTutorial VERSION 1.0.0 LANGUAGES CXX CUDA)

# ========== 设置标准 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_STANDARD 17) # RTX 4060

# ========== 查找需要的库 ==========
find_package(OpenGL REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)

# ========== 添加include路径 ==========
include_directories(include/)

# ========== Qt初始化 ==========
qt_standard_project_setup()

# ========== 源文件与头文件 ==========
set(CPP_SOURCES 
    src/ptScene.cpp
    tests/PathTracing.cpp)

set(HEADERS 
    include/ptScene.h
    include/Primitive.h
    include/Material.h)

set(CUDA_SOURCES 
    src/pathTracingKernel.cu)


# ========== 添加可执行目标 ==========
add_executable(PathTracing ${HEADERS} ${CPP_SOURCES} ${CUDA_SOURCES})

# ========== 链接库 ==========
target_link_libraries(PathTracing PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    OpenGL::GL
)