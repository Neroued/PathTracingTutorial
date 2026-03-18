cmake_minimum_required(VERSION 3.20)
project(PathTracer VERSION 2.0.0 LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ========== Standards ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ========== CUDA toolkit ==========
find_package(CUDAToolkit REQUIRED)

# ========== CUDA flags ==========
set(CMAKE_CUDA_FLAGS_DEBUG          "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -g -lineinfo")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE        "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# MSVC compatibility
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)

# ========== 3rdparty: tinyobjloader ==========
add_library(tinyobjloader STATIC 3rdparty/tinyobjloader/tinyObjLoader.cpp)
target_include_directories(tinyobjloader PUBLIC 3rdparty/tinyobjloader)

# ========== 3rdparty: nlohmann_json (header-only) ==========
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE 3rdparty/json)

# ========== 3rdparty: stb (implementation unit) ==========
add_library(stb_impl STATIC 3rdparty/stb/stb_impl.cpp)
target_include_directories(stb_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

# ========== 3rdparty: tinygltf (header-only) ==========
add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE 3rdparty/tinygltf)

# ========== Render core library ==========
set(PT_CORE_SOURCES
    src/io/image_io.cpp
    src/io/mesh_loader.cpp
    src/io/gltf_loader.cpp
    src/geometry/shapes.cpp
    src/accel/bvh_builder.cpp
    src/lights/light_sampler.cpp
    src/cuda/cuda_texture.cu
    src/scene/device_scene.cu
    src/scene/scene_loader.cpp
    src/scene/scene_manager.cpp
    src/film/film.cu
    src/integrators/path.cu
    src/renderer.cu
)

set(PT_CORE_HEADERS
    src/pt.h
    src/cuda/check.h
    src/cuda/cuda_buffer.h
    src/cuda/cuda_texture.h
    src/math/vecmath.h
    src/math/mat4.h
    src/math/math_utils.h
    src/math/bounds.h
    src/math/frame.h
    src/math/sampling.h
    src/core/ray.h
    src/core/interaction.h
    src/geometry/vertex.h
    src/geometry/triangle.h
    src/accel/bvh_common.h
    src/accel/bvh_builder.h
    src/accel/bvh_traversal.cuh
    src/materials/material.h
    src/materials/microfacet.cuh
    src/materials/disney_bsdf.cuh
    src/materials/shading_utils.cuh
    src/lights/envmap.h
    src/lights/light_sampler.h
    src/lights/punctual_light.h
    src/lights/light_pdf.cuh
    src/camera/camera.h
    src/samplers/pcg.cuh
    src/samplers/sobol.cuh
    src/samplers/sampler.cuh
    src/integrators/integrator.h
    src/integrators/path_integrator.h
    src/integrators/direct_lighting.cuh
    src/film/film.h
    src/scene/scene.h
    src/scene/device_scene.h
    src/scene/scene_manager.h
    src/io/image_io.h
    src/io/gltf_loader.h
)

add_library(pt_core STATIC ${PT_CORE_SOURCES} ${PT_CORE_HEADERS})

target_include_directories(pt_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
)

target_link_libraries(pt_core
    PUBLIC
        CUDA::cudart
        tinyobjloader
        nlohmann_json
        stb_impl
        tinygltf
)

# ========== CLI executable ==========
add_executable(pt src/main.cpp)
target_link_libraries(pt PRIVATE pt_core)

# ========== Tests ==========
enable_testing()

add_executable(pt_test_math tests/test_math.cpp)
target_link_libraries(pt_test_math PRIVATE pt_core)
add_test(NAME math_tests COMMAND pt_test_math)

add_executable(pt_test_bvh tests/test_bvh.cpp src/accel/bvh_builder.cpp)
target_link_libraries(pt_test_bvh PRIVATE pt_core)
add_test(NAME bvh_tests COMMAND pt_test_bvh)

add_executable(pt_test_sampling tests/test_sampling.cpp)
target_link_libraries(pt_test_sampling PRIVATE pt_core)
add_test(NAME sampling_tests COMMAND pt_test_sampling)

# ========== 3rdparty: GLAD (OpenGL loader) ==========
add_library(glad STATIC 3rdparty/glad/src/glad.c)
target_include_directories(glad PUBLIC 3rdparty/glad/include)

# ========== 3rdparty: GLFW ==========
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(3rdparty/glfw)

# ========== 3rdparty: Dear ImGui ==========
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw)

# ========== Preview executable ==========
option(PT_BUILD_PREVIEW "Build ImGui interactive preview" ON)
if(PT_BUILD_PREVIEW)
    find_package(OpenGL REQUIRED)

    set(PREVIEW_SOURCES
        preview/gl_display.cpp
        preview/imgui_layer.cpp
        preview/preview_window.cpp
    )

    add_executable(pt_preview src/main.cpp ${PREVIEW_SOURCES})
    target_include_directories(pt_preview PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/preview
    )
    target_link_libraries(pt_preview PRIVATE
        pt_core glad glfw imgui OpenGL::GL
    )
    target_compile_definitions(pt_preview PRIVATE PT_HAS_PREVIEW=1)

    # Copy shader files to build output
    add_custom_command(TARGET pt_preview POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:pt_preview>/preview
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/preview/fullscreen_quad.vert
            ${CMAKE_CURRENT_SOURCE_DIR}/preview/tone_mapping.frag
            $<TARGET_FILE_DIR:pt_preview>/preview/
    )
endif()
