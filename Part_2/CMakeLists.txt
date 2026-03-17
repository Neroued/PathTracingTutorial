cmake_minimum_required(VERSION 3.16)
project(PathTracingTutorial VERSION 1.0.0 LANGUAGES CXX CUDA)

# 生成 compile_commands.json，供 clangd/clang-tidy 使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ========== 设置标准 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120) # RTX 5090
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # 允许跨文件调用 __device__ 函数或 __constant__ 数据

# 启用 cuda 调试信息
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -lineinfo -Xptxas=\"-v\"")

# 按配置设置 CUDA 优化与调试标志，避免 Debug 下 /RTC1 与 /O2 冲突
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -g -lineinfo")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# ========== MSVC 与 Qt 配合需要设置 ==========
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)

# ========== 选择正确的 Qt 根目录（支持 QT_MSVC 或 QT_MINGW） ==========
if(DEFINED ENV{QT_MSVC})
    set(QT_ROOT "$ENV{QT_MSVC}")
elseif(DEFINED ENV{QT_MINGW})
    set(QT_ROOT "$ENV{QT_MINGW}")
else()
    message(FATAL_ERROR "请设置环境变量 QT_MSVC 或 QT_MINGW 指向 Qt 安装根目录")
endif()

# 让 CMake 能找到 Qt 安装
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${QT_ROOT}")

# ========== 查找需要的库 ==========
find_package(OpenGL REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)

# ========== 添加 HDRImageViewer ==========
add_subdirectory(HDRImageViewer)

# ========== 添加include路径 ==========
include_directories(include/)
include_directories($ENV{CUDA_PATH}/include)
include_directories(extern/)

# ========== Qt初始化 ==========
qt_standard_project_setup()
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ========== 源文件与头文件 ==========
set(CPP_SOURCES
    src/Scene.cpp
    src/BVH.cpp
    src/ImageLoader.cpp
    src/Texture.cpp
    src/MainWindow.cpp
    tests/PathTracing.cpp)

set(HEADERS
    include/Scene.h
    include/Material.h
    include/MainWindow.h)

set(CUDA_SOURCES
    src/pathTracingKernel.cu
    src/mixKernel.cu
    src/SceneConstants.cu)

set(CUDA_HEADERS
    include/random_pcg.cuh
    include/SceneConstants.cuh)

set(EXTERN_SOURCES
    extern/tinyObjLoader.cpp)

# ========== 添加可执行目标 ==========
add_executable(PathTracing ${HEADERS} ${CUDA_HEADERS})
target_sources(PathTracing PRIVATE ${CPP_SOURCES} ${EXTERN_SOURCES})
target_sources(PathTracing PRIVATE ${CUDA_SOURCES})

add_executable(testBVH tests/testBVH.cpp src/BVH.cpp)

add_executable(testTinyObj tests/testTinyObj.cpp ${EXTERN_SOURCES})

add_executable(testImage tests/testImage.cpp src/ImageLoader.cpp)

add_executable(testHDRImageViewer tests/testHDRImageViewer.cpp)

target_link_libraries(testHDRImageViewer PRIVATE HDRImageViewer)

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
# 选择 windeployqt 可执行文件（Qt6 为 windeployqt.exe 或 windeployqt6.exe）
set(WINDEPLOYQT "${QT_ROOT}/bin/windeployqt6.exe")
if(NOT EXISTS "${WINDEPLOYQT}")
    set(WINDEPLOYQT "${QT_ROOT}/bin/windeployqt.exe")
endif()

add_custom_command(TARGET PathTracing POST_BUILD
    COMMAND "${WINDEPLOYQT}" "$<TARGET_FILE:PathTracing>"
    COMMENT "Running windeployqt to deploy Qt libraries..."
)

# ========== 导出并复制 compile_commands.json 到源码根目录（便于 clangd 使用） ==========
# if(CMAKE_EXPORT_COMPILE_COMMANDS)
#     add_custom_target(copy_compile_commands ALL
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                 ${CMAKE_BINARY_DIR}/compile_commands.json
#                 ${CMAKE_SOURCE_DIR}/compile_commands.json
#         DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
#         COMMENT "Copying compile_commands.json to source directory"
#     )
# endif()
