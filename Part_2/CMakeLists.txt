cmake_minimum_required(VERSION 3.16)
project(PathTracingTutorial VERSION 1.0.0 LANGUAGES CXX CUDA)

# ========== 设置标准 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89) # RTX 4060
set(CMAKE_CUDA_STANDARD 17)

# ========== MSVC 与 Qt 配合需要设置 ==========
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)

# ========== 选择正确的Qt ==========
if(MSVC)
    message(STATUS "Using MSVC compiler")
    set(QT_DIR $ENV{QT_MSVC})
    set(CMAKE_PREFIX_PATH ${QT_DIR})
else()
    message(WARNING "Unknown compiler: please set CMAKE_PREFIX_PATH manually")
endif()

# ========== 查找需要的库 ==========
find_package(OpenGL REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)

# ========== 添加include路径 ==========
include_directories(include/)
include_directories($ENV{CUDA_PATH}/include)

# ========== Qt初始化 ==========
qt_standard_project_setup()

# ========== 源文件与头文件 ==========
set(CPP_SOURCES
    src/Scene.cpp
    tests/PathTracing.cpp)

set(HEADERS
    include/Scene.h
    include/Primitive.h
    include/Material.h)

set(CUDA_SOURCES
    src/pathTracingKernel.cu
    src/mixKernel.cu)

# ========== 添加可执行目标 ==========
add_executable(PathTracing ${HEADERS})
target_sources(PathTracing PRIVATE ${CPP_SOURCES})
target_sources(PathTracing PRIVATE ${CUDA_SOURCES})

# ========== 链接库 ==========
target_link_libraries(PathTracing PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    OpenGL::GL
)

# ========== 添加所需运行库 ==========
add_custom_command(TARGET PathTracing POST_BUILD
    COMMAND "${QT_DIR}/bin/windeployqt6.exe" "$<TARGET_FILE:PathTracing>"
    COMMENT "Running windeployqt6 to deploy Qt libraries..."
)

# ========== 修改compile_commands.json ==========
add_custom_target(transform_commands ALL
    COMMAND ${CMAKE_COMMAND} -E echo "transforming compile_commands.json ..."
    COMMAND python3 ${CMAKE_SOURCE_DIR}/utils/fix_clangd.py
    DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "transforming compile_commands.json"
)
