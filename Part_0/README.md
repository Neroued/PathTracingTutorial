# OpenGL 路径追踪 -- Part 0：开始前准备

![](.\img\pt.png)

## 一、前言

这个系列文章的主要目的是实现一个简单的路径追踪渲染器。这个系列适合希望入门路径追踪的人观看，因为笔者自己也只是刚刚开始学习，这也是笔者的学习笔记，用来查漏补缺。假如在文中任何地方有任何问题，还请读者不吝指教。在开始我的内容之前，先推荐一些对我本人帮助很大的文章。

- [LearnOpenGL](https://learnopengl.com/) 经典的OpenGL入门教程，若从未接触过OpenGL的api，可以从这里学习。
- AKGWSB 大佬的 [EzRT](https://github.com/AKGWSB/EzRT) 教程，详实且一步步推进，本系列之中很多内容都来自这位大佬的博客，如果读者认为我写的太烂，可以直接移步大佬的博客，那里会有更好的。
- [pbrt](https://github.com/mmp/pbrt-v4) 的官方代码仓库。图形学经典书籍的代码仓库，里面可以找到很多经典的算法实现。
- 以及我本人的代码仓库，这个系列的代码都会上传到这个仓库。
- 还有更多会在将来补充...

我选择OpenGL作为图形api，使用其在4.3版本之后提供的compute shader实现在GPU上加速计算。为了减少在代码封装与抽象上的工作量，我选择Qt作为框架，以下是我编写代码时的环境：

- Windows 11
- Qt 6.8.1
- mingw gcc 13.1.0 (Qt附带)
- c++ 17
- vscode
- cmake 3.29.3

同样的，读者也可以选择使用例如glfw/glew来加载OpenGL，使用glm作为数学运算库，不过这可能会需要一些额外的代码封装，但这并不困难。

接下来，正式进入本系列的第零篇文章，编写一个compute shader的Hello World。

## 二、概述

首先我们需要对程序的流程有一个大致的了解。可以根据运行设备的不同，将程序分为CPU部分和GPU部分。我们需要在CPU部分进行程序的初始化，包括编译着色器、创建场景、上传场景信息等内容。在GPU部分利用compute shader进行路径追踪的计算，再利用另一个render program进行计算结果的展示。流程可以简化为：程序启动 --> 编译着色器 --> 加载场景 --> 上传场景信息至GPU --> 计算着色器进行路径追踪计算 --> 渲染并展示结果。因此在CPU端，我们需要一个类，完成编译着色器与加载并上传场景信息的功能。

在GPU端，compute shader是独立于传统光栅化流程的一个pass，并不能直接展示计算的结果，我们仍然需要借助传统的顶点与片段着色器来展示。因此我们总体上需要两个shader program。一个是核心的用于计算的compute shader，一个是由简单的顶点和片段着色器构成的render program。在后文中会更加详细的介绍compute shader和传统光栅化流程的区别。

## 三、工程的创建

在这个部分，我们将使用vscode与cmake，从头开始创建我们路径追踪的工程。

这里我假设读者们已经安装好了上述环境，这篇文章不涉及安装相关的内容。

创建一个文件夹，命名为你喜欢的名字，例如`PathTracingTutorial`, 然后进入这个文件夹，创建五个目录，`src`, `include`, `extern`, `shaders`, `tests`，分别对应`.cpp源文件`，`.h头文件`，`引用的外部库`，`着色器代码`，`测试代码`。同时还需要一个`CMakeLists.txt` 文件。此时目录组织如下：

```
PathTracingTutorial/
├── extern/
├── include/
├── shaders/
├── src/
├── tests/
└── CMakeLists.txt
```

### 3.1 `CMakeLists.txt`

编辑`CMakeLists.txt`文件，写入以下内容：

```cmake
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
```

需要注意：Qt提供的OpenGL组件也类似于glfw，只不过进行了一些封装，底层仍然需要本机的OpenGL支持，因此需要链接进来。同时，Qt的信号与槽等机制需要头文件中的信息，因此头文件也需要被包含。

至此便完成了`CMakeLists.txt`文件的配置。

### 3.2 `ptScene`类创建

接下来创建一个`ptScene`类，作为路径追踪场景的核心。我们将在这个类中编译着色器，加载场景，并将场景信息上传至GPU。

分别在`include/`与`src/`下创建`ptScene.h`和`ptScene.cpp`文件。我们可以根据需求给出以下的声明:

```c++
class ptScene : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core
{
public:
    ptScene(QWidget *parent = nullptr);
    ~ptScene();
    void loadScene(); // 加载场景

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

private:
    void compileShaders(); // 编译所需的着色器

    void computeShaderPass(); // 计算pass，即进行路径追踪的pass
    void renderShaderPass();  // 渲染pass，即展示计算结果的pass
	
    void uploadScene(); // 上传场景信息
    void initializeQuad(); // 初始化全屏四边形
    void createTexture(GLuint *texture, int width, int height, GLuint unit); // 创建材质

private:
    QOpenGLShaderProgram *m_computeProgram; // 计算着色器
    QOpenGLShaderProgram *m_renderProgram;  // 展示结果的着色器
    
    QOpenGLVertexArrayObject m_screenVAO;  // 用于屏幕的vao和vbo
    QOpenGLBuffer m_screenVBO;
    GLuint m_computeTexture; // 存放计算结果的材质
    int m_width, m_height;
};
```

`ptScene`继承自`QOpenGLWidget`，核心的需要重写的三个函数为：

- `void initializeGL()` 在初始化或重建OpenGL上下文时调用的函数，在这里我们可以完成我们所需的初始化操作，即编译着色器，加载与上传场景信息。

- `void resizeGL(int w, int h)`在窗口尺寸发生变化时调用的函数。
- `void paintGL()`每一帧渲染时调用的函数，我们在这里调用所需的着色器进行计算与渲染操作。

### 3.3 着色器代码编写

要理解我们应该如何组织着色器，首先需要明白什么是计算着色器。建议读者前往阅读[LearnOpenGL上关于compute shader的介绍](https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction)，在这里我也给出我个人的理解。

计算着色器可以利用GPU在给定的内存区域上进行特定操作。计算着色器没有直接的输入与输出方式，需要通过例如创建SSBO(Shader Storage Buffer Object)的方式来读写数据。

![图片来自LearnOpenGL](.\img\global_work_groups.png)

计算着色器可以充分的利用GPU的并行计算能力。首先需要引入”工作组“的概念。工作组是GPU并行计算的基本组织单位，每个工作组内存在一系列线程。工作组本身的维度是三维的。它在X, Y, Z方向上都分别有一定数量的工作线程。将需要处理的数据视为一个三维的长方体，每个工作组处理长方体中的一个小长方体，因此总的工作组的维度也是三维的。

在我们的应用场景中，我们希望使用compute shader计算一个例如 800 x 800大小的图像，因为图像是二维的，我们可以将Z方向上的工作组数量以及工作组大小都设置为1。将工作组大小设置为 (16, 16, 1), 表示一个工作组负责图像中16x16的区域。我们可以将工作组的数量设置为 (50, 50, 1)，便能得到总大小与期望一致的工作线程数量。

同时我们还需要创建一个尺寸为 800 x 800，存储数据类型为RGBA32F的SSBO用来存储计算的结果。

由于计算着色器不能直接展示计算结果，我们还需要使用另外的顶点与片段着色器来渲染。这一部分我们只需要简单的输出结果便可。将上述存储了计算结果的SSBO作为一个sampler2D传递给片段着色器，简单采样并输出即可。

首先给出顶点着色器与片段着色器的代码

```glsl
// pt_vertex.glsl
#version 450 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 TexCoord;

void main()
{
    TexCoord = aTexCoord;
    gl_Position = vec4(aPos, 0.0, 1.0);
}


// pt_fragment.glsl
#version 450 core

in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D renderedTexture;

void main()
{
    vec3 texCol = texture(renderedTexture, TexCoord).rgb;

    // gamma矫正
    float gamma = 2.2;

    vec3 colorGamma;
    colorGamma.r = pow(texCol.r, 1.0f / gamma);
    colorGamma.g = pow(texCol.g, 1.0f / gamma);
    colorGamma.z = pow(texCol.z, 1.0f / gamma);

    FragColor = vec4(colorGamma, 1.0);
}
```

这里为了能够正确在屏幕上展示颜色，添加了伽马矫正。可以看出，这两个着色器只是简单的将传入的`renderedTexture`绘制占据整个窗口的矩形上（这个矩形我们在`void initializeQuad()`函数中初始化，具体实现可以查看仓库中的完整代码）。

然后我们创建用于存储计算结果的SSBO。

```glsl
void ptScene::createTexture(GLuint *texture, int width, int height, GLuint unit)
{
    if (*texture)
    {
        glDeleteTextures(1, texture);
        *texture = 0;
    }
    // 生成新的纹理
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);

    // 设置纹理参数
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // 分配纹理内存
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    // 将新纹理绑定到图像单元 unit
    glBindImageTexture(unit, *texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
}
```

我们使用

```glsl
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
```

保证了超出范围时不会发生错误。

接下来我们开始编写compute shader相关的内容。

在`computeShaderPass`中，我们调用计算着色器进行一次完整的计算过程。具体过程如下：

```c++
void ptScene::computeShaderPass()
{
    m_computeProgram->bind();

    // 计算工作组数，保证覆盖整个纹理区域（计算着色器中 local_size 为 16×16）
    GLuint groupX = (m_width + 15) / 16;
    GLuint groupY = (m_height + 15) / 16;
    glDispatchCompute(groupX, groupY, 1);

    // 内存屏障，确保计算着色器写入完成后 fragment shader 能正确采样
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    m_computeProgram->release();
}
```

首先分别计算X, Y方向上的工作组数量，保证能够覆盖整个纹理区域。

使用`glDispatchCompute(groupX, groupY, 1)`调用了大小为 (groupX, groupY, 1) 的工作组开始计算。在计算着色器中，我们设置工作组大小为 (16, 16, 1)。`glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)`来保证所有工作线程都完成计算。

在本章的最后，我们编写一个计算着色器的”Hello World"程序作为收尾。

```glsl
// pt_compute.glsl
#version 450 core

/* ----- 设置工作组大小 ----- */
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

/* ----- 输出的图像 ----- */
layout(rgba32f, binding = 0) uniform image2D imgOutput;

void main()
{
    vec4 value = vec4(0.0, 0.0, 0.0, 1.0);
    // 当前线程对应的像素的坐标，范围是[0, width] x [0, height]
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    
    // 将数值根据总的工作线程数量归一化
    value.x = float(texelCoord.x)/(gl_NumWorkGroups.x * gl_WorkGroupSize.x);
    value.y = float(texelCoord.y)/(gl_NumWorkGroups.y * gl_WorkGroupSize.y);

    imageStore(imgOutput, texelCoord, value);
}
```

我们会得到这样的输出：

![](.\img\helloworld.png)

## 四、总结

至此我们完成了基本框架的搭建，并成功输出了由计算着色器计算的图像。下一部分我们将在这个框架的基础上完善加载场景的功能，并尝试将场景直接展示出来。

