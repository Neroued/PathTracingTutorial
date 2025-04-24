# Windows 平台路径追踪渲染器架构设计

## 项目概述与需求分析

本项目需要设计一个运行于 Windows 平台的**路径追踪渲染器**，界面基于 Qt，渲染核心使用 CUDA 实现。系统需满足以下功能和要求：

- **图形界面与显示**：使用 Qt 提供交互界面，包含一个使用 `QOpenGLWidget` 的 OpenGL 显示区域，用于显示渲染结果纹理。用户可以切换“逐帧渲染”或“持续迭代渲染”模式。
- **参数调整与渲染重启**：当用户修改渲染参数（如采样数、光线反弹次数等）时，需要**重新开始渲染**。而**实时显示参数**（如色调映射曲线和曝光）调整不应打断渲染过程。
- **HDR 输出与色调映射**：渲染输出为高动态范围（HDR）图像，因此在显示时需要进行**色调映射（tone mapping）**和曝光调整。用户应能在界面中实时控制这些显示参数，而不必重启渲染器。显示过程中将HDR像素转换为低动态范围以适配显示器 ([LearnOpenGL - HDR](https://learnopengl.com/Advanced-Lighting/HDR#:~:text=High dynamic range rendering works,favors dark or bright regions))。
- **渲染核心**：采用自定义的路径追踪实现（不依赖第三方渲染库），充分利用CUDA在GPU上计算。渲染器与界面应**解耦**，通过异步通信（信号/回调）交互，使渲染在后台线程运行，界面在主线程响应用户输入。
- **OpenGL-CUDA 数据交换**：需要研究两种GPU数据交换方式并权衡：① 将 OpenGL 纹理直接注册为 CUDA 可访问的表面（surface object），CUDA 内核直接写入纹理；② CUDA 渲染至独立缓冲区（如 PBO），完成后通过 OpenGL 将数据上传到纹理。需比较二者在性能、灵活性、稳定性方面的优劣，并选择适合本项目的方案。

下面将针对这些方面进行详细设计和说明。

## 界面框架选择：Qt Widgets vs Qt Quick

Qt 提供了两种主要的界面技术：传统的 **Qt Widgets**（基于 C++ 的组件）和 **Qt Quick/QML**（基于声明式 QML 的界面）。本项目需要在桌面环境下提供包含OpenGL视窗和参数控制的GUI，我们在选择时需要考虑开发成本与最终效果之间的平衡。

- **Qt Widgets 优势**：Qt Widgets 是成熟的桌面 UI 库，采用 C++ 实现，与底层操作系统 API 紧密结合，具有**高性能和稳定性**，非常适合复杂计算和数据密集型的应用 ([简单介绍Qt Quick、QML以及Qt Widgets之间的区别与联系_qquickwidget-CSDN博客](https://blog.csdn.net/Jacksqh/article/details/130703212#:~:text=))。对于本项目而言，渲染计算主要在后台CUDA进行，Widgets能够轻松创建所需的窗口、小部件（按钮、滑块等）来控制参数，并且**学习成本低**（很多Qt C++开发者已非常熟悉 Widgets）。在需要处理大量数据或复杂运算时，Qt Widgets 通常比QML更快 ([简单介绍Qt Quick、QML以及Qt Widgets之间的区别与联系_qquickwidget-CSDN博客](https://blog.csdn.net/Jacksqh/article/details/130703212#:~:text=))。此外，Qt Widgets 提供原生的窗口和控件外观，系统集成良好。
- **Qt Quick/QML 优势**：Qt Quick 使用 QML 声明式语言描述界面，适合构建现代化、动画丰富的UI。如果需要非常华丽和动态的界面效果，Qt Quick 更有优势，能够充分利用GPU加速绘制，UI动画流畅。此外，Qt Quick 可以方便地实现跨平台统一的界面风格，不依赖各平台的原生控件外观 ([简单介绍Qt Quick、QML以及Qt Widgets之间的区别与联系_qquickwidget-CSDN博客](https://blog.csdn.net/Jacksqh/article/details/130703212#:~:text=通读全文，我们可以看出，Qt Widgets有很多的优点，比如稳定、性能好等等优点，所以对于大规模的应用程序来说还是使用Qt Widgets来得靠谱，但是对于UI界面来说，想要单独使用Qt Widgets设计好美观和炫酷的UI界面是非常困难的，所以这才要引入QML，而Qt Quick是QML的一个框架可以更好地使用QML，但是QML的性能和稳定性是没有Qt,Widgets好的，所以Qt Widgets与（QML、Qt Quick）也算是互补，将它们进行结合可以让我们的程序更完美。))。QML语法高层次、易于描述复杂界面布局，适合UI与逻辑分离。
- **劣势与权衡**：Qt Quick/QML 的学习成本相对较高，需要熟悉QML/JavaScript语法以及与C++交互，调试也相对复杂。而 Qt Widgets 对于已有C++/Qt经验的开发者更直接。QML在性能和稳定性方面相对Widgets略逊一筹，特别是在大量数据更新的场景下 ([简单介绍Qt Quick、QML以及Qt Widgets之间的区别与联系_qquickwidget-CSDN博客](https://blog.csdn.net/Jacksqh/article/details/130703212#:~:text=通读全文，我们可以看出，Qt Widgets有很多的优点，比如稳定、性能好等等优点，所以对于大规模的应用程序来说还是使用Qt Widgets来得靠谱，但是对于UI界面来说，想要单独使用Qt Widgets设计好美观和炫酷的UI界面是非常困难的，所以这才要引入QML，而Qt Quick是QML的一个框架可以更好地使用QML，但是QML的性能和稳定性是没有Qt,Widgets好的，所以Qt Widgets与（QML、Qt Quick）也算是互补，将它们进行结合可以让我们的程序更完美。))。鉴于本项目主要功能是渲染展示和参数调整，界面相对传统（几个按钮和滑杆，不需要复杂动画），采用 Qt Widgets 更为稳妥高效。此外，可以在Qt Widgets中嵌入 QOpenGLWidget 来利用GPU显示，加之渲染负载主要在CUDA上，Qt Widgets完全可以胜任。因此，在**开发成本**和**性能稳定性**的考量下，**推荐使用 Qt Widgets** 实现界面。Qt Quick 可在未来有富余精力时再考虑，用于优化界面美观度。总之，本项目优先注重渲染性能和可靠性，Qt Widgets 是一个稳健的选择 ([简单介绍Qt Quick、QML以及Qt Widgets之间的区别与联系_qquickwidget-CSDN博客](https://blog.csdn.net/Jacksqh/article/details/130703212#:~:text=通读全文，我们可以看出，Qt Widgets有很多的优点，比如稳定、性能好等等优点，所以对于大规模的应用程序来说还是使用Qt Widgets来得靠谱，但是对于UI界面来说，想要单独使用Qt Widgets设计好美观和炫酷的UI界面是非常困难的，所以这才要引入QML，而Qt Quick是QML的一个框架可以更好地使用QML，但是QML的性能和稳定性是没有Qt,Widgets好的，所以Qt Widgets与（QML、Qt Quick）也算是互补，将它们进行结合可以让我们的程序更完美。))。

*(注：Qt Widgets 和 Qt Quick 也可混合使用，例如采用 Widgets 主窗口嵌入 Qt Quick 界面或使用 Qt Quick 的 QQuickWidget。但为避免复杂性，此处推荐单一框架。)*

## 渲染与界面解耦（多线程架构）

为了保证UI响应流畅，渲染计算不阻塞界面，需要将**渲染器与界面运行在不同线程**。具体而言，我们将在**后台线程**运行CUDA路径追踪渲染循环，在**GUI主线程**运行Qt界面和OpenGL显示。界面和渲染器之间通过信号/槽或回调异步通信，实现解耦。

这一设计符合通用实践：正如有人在讨论**渐进式路径追踪**时提到的，“将渲染器置于单独线程，逐步将部分结果发送到主线程，主线程创建窗口及时显示结果” ([opengl - Simple Progressive Rendering Window for a Path Tracer - Stack Overflow](https://stackoverflow.com/questions/39620498/simple-progressive-rendering-window-for-a-path-tracer#:~:text=It is not restricted to,No magic here))。没有魔法，实现原理很简单：**后台线程持续渲染->主线程持续更新显示**。这样的架构确保了渲染计算再密集，也不会卡死UI，用户仍可随时调整参数或停止渲染 ([opengl - Simple Progressive Rendering Window for a Path Tracer - Stack Overflow](https://stackoverflow.com/questions/39620498/simple-progressive-rendering-window-for-a-path-tracer#:~:text=It is not restricted to,No magic here))。

- **后台渲染线程**：我们可以创建一个继承自 `QThread` 的 `RenderThread` 类，用于执行CUDA渲染。在线程的 `run()` 方法中启动路径追踪的主循环：根据模式选择单次渲染或持续迭代渲染。该线程内部维护渲染状态（如当前样本数、累积帧缓冲等）并与CUDA交互。由于是独立线程，我们可以在其中自由调用CUDA kernel，而不会阻塞GUI。
- **主界面线程**：运行 Qt 事件循环，负责处理用户交互和绘制界面。渲染线程产生的新帧数据需要传递到主线程进行OpenGL纹理更新和显示。这通过发送信号来实现——渲染线程在每帧完成或每次迭代后，发出一个信号（如 `frameReady`），将控制权交回主线程。主线程的槽函数捕获该信号，随后触发OpenGL控件刷新显示新帧。
- **信号与线程安全**：Qt的信号槽机制在跨线程连接时默认使用**Queued Connection**，这意味着信号发射时参数被拷贝到事件队列，接收槽在目标线程的事件循环中执行，从而实现线程间安全通信。我们可以定义例如：`RenderThread::frameReady()` 信号（不携带图像数据，仅通知），连接到界面 `GLViewport` 对象的更新槽；以及 `RenderThread::finished()` 信号连接到界面槽以处理渲染完成状态。这样避免了共享数据的竞争条件。当渲染线程需要接受参数变更时，也可以通过线程安全的方法（如将新参数包裹到`std::atomic`或发信号）通知它。在简单实现中，我们也可以每次参数改变时**重启渲染线程**，从而天然避免旧线程与新参数竞争。
- **渲染循环控制**：在“持续迭代渲染”模式下，渲染线程不会自行结束，而是不断生成新样本帧，直到用户停止或参数改变。在“单帧渲染”模式下（例如只渲染固定采样数达到目标质量就停止），渲染线程在完成后可以退出或等待进一步指令。两种模式可以通过线程中的一个标志或状态机控制。UI上提供一个开关来选择模式，内部通过信号告知渲染线程调整行为。

## OpenGL 显示与逐帧更新机制

主线程中的 OpenGL 显示区域将使用 **QOpenGLWidget** 实现。这个 widget 内部有自己的 OpenGL 上下文，用于绘制渲染结果纹理。显示的基本流程是：**将渲染输出作为纹理上传GPU -> 在QOpenGLWidget中绘制一个全屏矩形（quad）贴上该纹理**，从而把图像显示出来。关键点如下：

- **QOpenGLWidget 设置**：在 `QOpenGLWidget::initializeGL()` 中创建用于显示的**OpenGL纹理对象**（如 `GLuint outputTexture`）。这个纹理尺寸与渲染输出帧缓冲相同，内部格式使用浮点格式（例如 `GL_RGBA32F`）以承载HDR数据。纹理创建后，可以使用OpenGL函数或CUDA-OpenGL互操作机制将渲染数据传入纹理。在 `paintGL()` 中，我们绘制一个覆盖整个widget的矩形，并在片段着色器中采样此纹理。为了简化，可以使用Qt提供的 `QOpenGLFunctions` 绘制，或自己编写简单的着色器进行纹理显示。

- **逐帧/迭代更新**：每当渲染线程准备好新的帧数据，会发信号通知主线程。主线程接收到信号后，调用 `QOpenGLWidget::update()` 请求重绘。Qt随后会调用 `paintEvent` -> `paintGL()`. 在重绘时，我们需要确保OpenGL纹理包含了渲染线程最新输出的数据。在架构上，有两种实现途径：

  1. **拉取式**：主线程重绘时主动向渲染线程/缓冲获取当前图像数据上传纹理。这可以在 `paintGL()` 开始时执行，例如从共享内存拷贝像素到纹理。
  2. **推送式**：渲染线程每产生新结果就直接通过OpenGL API或CUDA-OpenGL互操作把纹理数据更新好，然后仅通知UI去绘制。这样 `paintGL()` 中无需再次传输数据，只需绘制已更新的纹理即可。

  本设计倾向第二种——**渲染线程在后台更新纹理数据，UI线程只是渲染显示**。因此渲染线程每次计算完新的帧/迭代时，可以直接写入GPU纹理（详见下一节的数据交换），然后发出信号。UI线程的 `paintGL()` 直接使用现有纹理绘制。这样避免在主线程执行繁重的数据拷贝，优化性能。

- **线程上下文注意**：默认情况下，所有 OpenGL 调用（如纹理更新、绘制）应在拥有该OpenGL上下文的线程执行。QOpenGLWidget的上下文附属在主线程GUI上。因此若后台线程需要直接操作纹理（例如通过CUDA映射的指针写入），需要确保同步和上下文共享。幸运的是，CUDA的OpenGL互操作并不要求在OpenGL上下文线程中调用CUDA函数，但OpenGL本身的调用（如 `glTexSubImage2D`）必须在主线程。因此我们采用 **CUDA 在后台写纹理 / 主线程仅绘制** 的模式：CUDA写入时通过特殊互操作接口，不使用OpenGL调用，从而避开线程不安全的问题。完成写入后，主线程按照正常绘制流程用该纹理渲染即可。

- **双缓冲考虑**：为防止渲染线程写入纹理与主线程读取纹理发生竞争，可以采用双缓冲技术。即准备两个 OpenGL 纹理（或缓冲），渲染线程每次写入不在显示的那个纹理，然后与主线程交换。但由于CUDA和OpenGL会通过映射/同步确保写读完整性，简单实现下单纹理亦可行。如果发现卡顿，可以扩展为双纹理交替更新显示，以进一步提高并发效率。

## HDR 输出的色调映射与曝光控制

渲染器输出的是HDR图像（高动态范围，像素可能超过[0,1]范围）。直接显示在标准显示设备上会导致高亮部分裁剪饱和。为此需要进行**色调映射**（Tone Mapping），将HDR颜色映射为可显示的LDR颜色，同时提供**曝光（Exposure）**调整以控制整体亮度 ([LearnOpenGL - HDR](https://learnopengl.com/Advanced-Lighting/HDR#:~:text=High dynamic range rendering works,favors dark or bright regions))。本项目对色调映射和曝光的处理方式如下：

- **渲染器输出保持HDR**：CUDA渲染线程将生成的图像保存在高精度缓冲中（例如每通道32位浮点）。在逐次迭代采样时，累积计算HDR像素值。渲染线程不对颜色做截断或Gamma修正，仅累积真实的物理亮度值。这样能够保留场景的完整动态范围信息。
- **色调映射在显示阶段完成**：我们在 OpenGL 显示阶段应用色调映射算法，将HDR像素转换为LDR。典型做法是在片段着色器中对取样到的HDR颜色执行色调映射函数。例如常用的**Reinhard色调映射**、**曝光调整**乘以`2^exposure`、Gamma校正等 ([LearnOpenGL - HDR](https://learnopengl.com/Advanced-Lighting/HDR#:~:text=High dynamic range rendering works,favors dark or bright regions))。由于希望用户实时调节，我们将**曝光值**和可能的色调映射曲线参数做成**Uniform**传入着色器，由UI控制改变uniform值即可。这样每次用户移动曝光滑块，只需调用一次 `update()` 触发重绘，新的曝光值会影响片段着色器输出，实现实时预览，无需重启渲染或重新计算HDR数据。
- **曝光和色调映射参数 UI**：界面上可以提供**曝光滑块**（模拟相机曝光，范围例如 -5 到 +5 stops）以及**色调映射下拉选择**或参数调整。如本项目不需要复杂选项，也可只提供曝光基本控制，内部使用固定的Reinhard等色调映射。用户调整这些参数时，我们**不重启渲染线程**，仅在显示层改变图像表现。这满足了要求：“展示参数的调整不需要重启渲染器”，因为渲染核心仍在持续采样，HDR累积结果未变，只是通过不同曝光/映射以LDR形式显示。
- **实现方式**：在 `QOpenGLWidget::initializeGL()` 中创建并编译一个简单的片段着色器，其中包含：`uniform sampler2D hdrTexture; uniform float exposure;` 等，然后在`gl_FragColor`计算时，对从hdrTexture取出的颜色执行诸如：`vec3 mapped = vec3(1.0) - exp(-color * pow(2.0, exposure));`（这是一种曝光和色调映射结合的公式）等。这样每次绘制都按照当前uniform计算。UI通过`QSlider::valueChanged`连接到一个槽，该槽使用 `openglWidget->makeCurrent();` 然后调用 `glUniform1f(exposureLoc, newExposure)` 更新uniform，再调用 `openglWidget->update();` 请求重绘。由于只是更新shader参数，开销很小，可以做到实时交互。

([LearnOpenGL - HDR](https://learnopengl.com/Advanced-Lighting/HDR#:~:text=High dynamic range rendering works,favors dark or bright regions)) 提到，不同曝光下同一HDR图像会显示出不同细节：低曝光看清高亮处细节，高曝光看清暗部细节。这正是我们提供曝光调节的意义。在实现中，可以选择合适的色调映射算法以兼顾亮部和暗部细节。简单起见，可用线性乘以曝光后再套用Gamma校正，或使用Reinhard全局色调映射等。

- **无需重启渲染**：强调一下，曝光和色调映射仅影响显示，不影响渲染过程。渲染线程始终输出HDR值。例如某像素HDR累计值非常亮（10.0），最初曝光=0时可能映射结果接近白色；如果用户调低曝光到-2，相当于乘0.25，再映射可能呈现出颜色细节。这整个过程中渲染线程毫不知情，继续计算其HDR值。一旦渲染进一步完善，该像素HDR值变化，显示结果也会动态更新。由于我们每次重绘都会重新从hdr纹理取样并色调映射，逐渐累积的改进会体现在显示上。所以架构上保持这种解耦非常重要。

## OpenGL-CUDA 数据交换方式分析

CUDA 渲染线程计算得到的图像数据，需要高效传输给 OpenGL 才能显示。有两种主要方案可以实现 CUDA 与 OpenGL 间的数据交换：

**方法一：OpenGL 纹理映射为 CUDA Surface，直接写入**
 在此方案中，我们先在主线程创建OpenGL纹理，然后使用 `cudaGraphicsGLRegisterImage` 将该纹理注册给CUDA ([ OpenGL Interoperability with CUDA | 3D Game Engine Programming	](https://www.3dgep.com/opengl-interoperability-with-cuda/#:~:text=To register an OpenGL texture,texture reference in CUDA later))。这样CUDA就获得了对这个 OpenGL纹理的访问权限，可在CUDA内核中将其当作 `surface<Object>` 来读写。步骤包括：

1. **注册纹理**：`cudaGraphicsGLRegisterImage(&resource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore)`. 注册后获得 `cudaGraphicsResource` 句柄。
2. **CUDA写纹理**：在每次需要更新时，渲染线程执行 `cudaGraphicsMapResources` 将纹理映射为CUDA内存，接着通过 `cudaGraphicsSubResourceGetMappedArray` 获取指向纹理数据的cudaArray，然后创建surface object绑定此数组。CUDA内核通过 `surf2Dwrite` 等直接对纹理像素写值。写完后调用 `cudaGraphicsUnmapResources` 解映射。完成后OpenGL纹理内容已更新。主线程在随后的绘制中使用纹理即可。

该方法的**优点**是避免了CPU内存拷贝，数据始终留在GPU上直接由CUDA写入OpenGL纹理，实现零拷贝高效传输。理论上，这减少了一次GPU到GPU的数据复制，应该是最快的方式。同时，由于直接写入最终显示纹理，流程简单。

但是方法一也有**缺点和挑战**：

- **每帧映射开销**：每次 Map/Unmap OpenGL 资源本质上需要同步 GPU 操作，可能造成 pipeline 停顿。如果每帧都这样做，开销可能较大。在实际测试中，有人发现使用 surface 直写虽省去拷贝，但每帧 map/unmap 反而导致整体帧率降低很多 ([OpenGL & CUDA interop with surfaces slow... - CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/opengl-cuda-interop-with-surfaces-slow/62930#:~:text=in a loop ,unmap))。例如一位开发者尝试将每帧CPU拷贝改为CUDA surface直写，却惊讶地发现后者竟慢7倍 ([OpenGL & CUDA interop with surfaces slow... - CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/opengl-cuda-interop-with-surfaces-slow/62930#:~:text=in a loop ,unmap))！这提示我们，**驱动同步开销**不可忽视。优化的方法是在可能情况下**减少map/unmap调用频率**。例如可以考虑一次映射后连续写多帧（不过OpenGL需要知道何时更新）或者采用双缓冲纹理交替映射。
- **实现复杂度**：需要编写CUDA surface write代码，注意处理纹理边界、格式匹配等。如果需要读取纹理旧值进行累积，还需surface load。但 surface load/store 只在支持的架构可用（CC 2.x 以上支持，现代N卡一般没问题）。
- **灵活性**：直接写纹理意味着CUDA输出格式必须和OpenGL纹理格式严格匹配。如我们使用GL_RGBA32F，则CUDA这边也要用float4写。如果日后想换格式或部分更新，操作起来会比较低级繁琐。另外，若想在CUDA结果基础上再做进一步处理（比如在CPU端读回一部分调试），直接纹理不方便读回（需要额外`glGetTexImage`等）。

**方法二：CUDA输出到独立缓冲，再通过OpenGL上传纹理**
 此方案使用一个独立的帧缓冲作为中转：CUDA内核将结果写入一个GPU内存缓冲区（可以是CUDA malloc的内存，或OpenGL Pixel Buffer Object），然后在OpenGL端将该缓冲的数据传给纹理。两种具体实现：

- **PBO (Pixel Buffer Object) 共享**：创建一个 OpenGL PBO（像素缓冲对象）作为纹理的数据源，将其注册给CUDA（使用 `cudaGraphicsGLRegisterBuffer`）。每帧CUDA将渲染结果写入这个PBO（通过映射获得其指针后 `cudaMemcpy` 或 kernel 写入）。写完后，主线程在绘制前执行 `glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO); glTexSubImage2D(..., NULL); glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);` 将PBO数据更新到纹理。这种方式下，`glTexSubImage2D` 从 PBO 拷贝到纹理的过程由GPU直接执行，避免了CPU拷贝。因为PBO驻留GPU，实质是一次GPU->GPU的DMA拷贝，由显卡驱动优化管理。
- **CUDA 内存 + glTexSubImage**：CUDA渲染到自己分配的内存（`cudaMalloc`的buffer）。每帧把该buffer拷到CPU内存（`cudaMemcpy DtoH`），再调用 `glTexSubImage2D` 用CPU指针提供数据上传到纹理。这其实是传统方案，但由于涉及**GPU到CPU再到GPU**，开销大，不推荐在实时应用中采用——仅在没有共享内存机制时可用。

对本项目，我们主要考虑前一种（PBO法）以避免CPU瓶颈。方法二的**优点**在于：

- **实现相对简单直观**：使用 PBO 方法，我们仍然在CUDA侧按常规操作内存，无需特殊的surface写语义。调试和维护也容易，例如可以在CUDA内核把结果写到一个float数组，然后一次memcpy到PBO。
- **较小的同步开销**：OpenGL的 `glTexSubImage2D` 调用通常会异步执行，当使用PBO时更是如此：驱动可以在后台调度GPU完成从PBO到纹理的复制，不一定阻塞CPU。我们也可以通过双PBO交替，实现在CUDA写入下一帧的同时，上一次的PBO数据正在传纹理，从而pipeline并行。
- **灵活性高**：有了独立缓冲，容易在CUDA中更改分辨率、做多级处理（比如先HDR累积再另一个kernel做色调映射到LDR缓冲）然后上传。纹理更新由我们手动触发，可以控制节奏。还可以方便地读取该缓冲用于保存图像到硬盘等。

方法二的**缺点**主要是比方法一**多了一次显存内存拷贝**（从CUDA缓冲到纹理）。然而在现代PCIe和GPU内部带宽下，这次拷贝通常不是瓶颈，尤其相对于路径追踪巨大的计算量来说，可以忽略不计。此外，通过PBO这个拷贝仍发生在GPU，不消耗CPU时间。

**性能对比**：综合考虑，若实现良好，方法一理论上最优（零拷贝），但实际上可能被每帧同步开销抵消甚至更慢 ([OpenGL & CUDA interop with surfaces slow... - CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/opengl-cuda-interop-with-surfaces-slow/62930#:~:text=in a loop ,unmap))。方法二多一道GPU内存复制，但可以更好地流水并行和优化驱动调度。在渲染计算占主导的大背景下，这点开销影响很小。除非我们追求极限每秒上百帧的传输，否则方法二的性能是可以接受的。

**稳定性和兼容性**：方法二依赖的技术更简单成熟——PBO是OpenGL标准，兼容广泛，驱动支持良好；CUDA对GL buffer的映射也非常常用。而方法一涉及surface write这种相对进阶特性，可能在某些驱动或GPU上出现Bug（需要特定驱动版本支持才能确保性能）。基于稳妥起见，**方法二更为可取**。

**推荐方案**：针对本项目，倾向采用**CUDA -> PBO -> OpenGL纹理**的传输方式，即上述方法二的 PBO 实现。它在性能上足够高效，并且代码实现和调试更简单。同时如果后续需要，可以比较容易地实现双缓冲机制提升帧率平稳性。而方法一尽管也可行，但需要仔细处理同步，可留作优化选项。例如，待项目基本跑通后，可尝试用CUDA直接写纹理并测量性能，与PBO方案比较，再决定是否切换。

*(补充：实际上NVIDIA官方文档 ([ OpenGL Interoperability with CUDA | 3D Game Engine Programming	](https://www.3dgep.com/opengl-interoperability-with-cuda/#:~:text=reading from and writing to,instead of a texture reference))也提到，若只是单向写入，可以用 `cudaGraphicsRegisterFlagsWriteDiscard` 提示不要保留旧数据，以优化性能。PBO方案相当于这种思路：每帧丢弃旧buffer重写。)*

## 程序架构设计

综合以上考虑，下面给出本渲染器程序的模块与类设计，以及各部分交互流程。重点强调**渲染器与界面的解耦**、**数据流向**和**参数交互**机制。

([image](https://chatgpt.com/c/67f855da-52f4-800c-8f59-c553d202b07b)) *路径追踪渲染器程序架构模块图：后台渲染线程通过共享HDR帧缓冲将结果传递给主线程的OpenGL显示；控制面板的参数调整通过信号影响渲染线程或OpenGL显示。绿色虚线表示用户对UI控件的操作触发的指令流（曝光等显示参数直接作用于显示线程，渲染参数传递给渲染线程），橙色虚线表示渲染线程向UI发送的更新通知，蓝色实线表示HDR像素数据在GPU内存中的流动。 ([opengl - Simple Progressive Rendering Window for a Path Tracer - Stack Overflow](https://stackoverflow.com/questions/39620498/simple-progressive-rendering-window-for-a-path-tracer#:~:text=It is not restricted to,No magic here)) ([OpenGL & CUDA interop with surfaces slow... - CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/opengl-cuda-interop-with-surfaces-slow/62930#:~:text=in a loop ,unmap))*

### 模块与类说明

- **`MainWindow` 主窗口**：Qt Widgets 主窗口，持有控制面板和GL显示控件。例如布局上使用水平拆分器左侧放控制面板，右侧放 QOpenGLWidget 显示区。MainWindow 负责将控制面板的用户操作连接到渲染线程和显示控件。例如，按下“开始渲染”按钮启动渲染线程；调整曝光滑杆调用 OpenGL 显示控件的接口更新曝光；修改采样数等参数则发送信号给渲染线程。
- **`ControlPanel` 控制面板**：可由多个QWidget构成，比如若干QSlider、QComboBox等，用于调整**tone mapping参数**（曝光值、色调映射模式）和**渲染参数**（如采样率、反弹深度、是否持续渲染等）。ControlPanel上的控件通过 Qt 的信号槽与 MainWindow/RenderThread/GLViewport沟通：
  - 曝光滑杆的 `valueChanged(double)` 信号连接到 `GLViewport` 的槽（调整曝光uniform并触发重绘）。
  - 色调映射模式下拉选择连接到 `GLViewport`，切换不同片元着色器算法或Uniform参数。
  - 渲染参数（如“最大采样数”spinbox）改变时，发送信号给 RenderThread（若渲染正在进行，则通知需要重启；或在下次开始时读取新值）。
  - “开始/停止渲染”按钮：若未运行则启动RenderThread线程运行渲染；若正在运行则请求停止当前渲染线程。
- **`GLViewport` 显示控件**：继承自 `QOpenGLWidget`，负责OpenGL初始化和绘制。内部持有：输出纹理ID、着色器程序、用于调节的uniform位置等。提供方法如 `setExposure(float)`、`setToneMappingCurve(int)` 供控制面板调用。这些方法会设置内部状态并调用 `update()` 重绘。`GLViewport::initializeGL` 中完成FBO/纹理等创建和CUDA注册（如果采用互操作）；`paintGL` 中绘制满屏四边形并应用当前曝光/色调映射。该类还可以连接渲染线程的信号，例如渲染线程发出 `frameReady()` 信号时，连接到它的一个槽来调用自己的 `update()`，以便及时刷新显示最新纹理。
- **`RenderThread` 渲染线程**：继承自 `QThread`，封装CUDA渲染逻辑。其核心是在 `run()` 中根据设置执行路径追踪：
  - 如果是持续渲染模式：进入一个循环，不断地为每个像素投射样本、累积结果。在每轮样本完毕或隔N轮，调用CUDA/GL数据交换更新纹理，然后通过 `emit frameReady()` 通知UI线程。如此循环直到收到停止信号或达到某条件。
  - 如果是单帧模式：根据参数执行固定采样数的循环，完成后发出 `finished()` 信号并退出线程循环。
     渲染线程可以维护一个HDR累积buffer（float数组，大小width*height）。每次迭代计算新样本并累加，然后将累积均值写入显示缓冲。**线程通信**：当控制面板修改渲染参数时，可能通过调用 `RenderThread::requestRestart(newParams)` 来请求线程重启。这可设置一个原子标志，让渲染循环 break，然后线程结束run或者重置状态再继续。如果停止渲染，则设置一个停止标志或直接调用 QThread::quit()/wait 退出线程。RenderThread 需要与 OpenGL 共享资源：初始化时可接受 `outputTexture` 或 PBO 资源指针，供CUDA注册用。
- **`PathTracer` 渲染器核心**（可选拆分）：将具体的路径追踪算法和数据结构实现封装起来。例如 `PathTracer` 类提供 `renderOneSample(float* hdrBuffer)` 接口，由 RenderThread 调用。这样可以将多线程逻辑与渲染算法解耦，方便独立开发测试路径追踪算法。同时PathTracer可以管理场景数据、加速结构等。由于问题重点不在场景管理，此处略去细节。

### 数据流与交互过程

1. **启动渲染**：用户在控制面板点击“开始”。MainWindow捕获此事件，读取当前参数（分辨率、模式等），创建并启动 RenderThread。在启动前，需确保 OpenGL 纹理/PBO 已创建并注册给CUDA（可以在 RenderThread 初次运行时由主线程准备好，或让 RenderThread 在自己的 context 中创建共享资源）。RenderThread 启动后，在后台GPU开始路径追踪计算。
2. **持续渲染更新**：假设在持续迭代模式，RenderThread 每完成一定量采样后（例如每新增1 spp或每经过X毫秒），执行以下操作：
   - 将当前累积HDR帧数据写入 OpenGL 可显示的缓冲/纹理中（通过CUDA-OpenGL互操作）。如采用PBO方案，则CUDA kernel将HDR浮点帧数据复制到PBO映射内存，然后主线程那边在重绘时更新纹理。如果采用直接surface写，则此时CUDA直接写纹理。
   - 发出信号 `frameReady()` (带比如当前已累积样本数等信息)。该信号通过Qt槽连接，通知主线程的 GLViewport 执行刷新显示。
   - 渲染线程继续下一轮采样计算，除非在此期间收到停止或参数改变请求。
3. **界面刷新**：主线程响应到 `frameReady()` 信号后，调用 `GLViewport->update()` 请求重绘。Qt随后调用 `paintGL()`：此时新的帧数据已经通过GPU缓冲更新到了`outputTexture`。paintGL 切换至GL上下文，执行着色器绘制 full-screen quad，将该纹理绘制到窗口。片元着色器对每个像素进行tone mapping和曝光调整后输出到默认帧缓冲，最终显示在窗口上。由于我们连续迭代，用户会看到图像由初始的噪声逐渐收敛变清晰，达到进度可视化的效果。
4. **参数调整**：在渲染过程中，用户可以调整控制面板上的参数：
   - **曝光调整**：滑动曝光滑杆时，直接影响 GLViewport 的 `exposure` uniform。比如滑动触发槽函数：`glViewport->setExposure(newExp)`，内部保存值并调用 `update()` 触发一次重绘。下一帧显示时，片元着色器应用了新的曝光，因此亮度立即变化。这不会干扰后台渲染线程，其HDR缓冲数据不受影响。
   - **暂停/恢复**：如实现此功能，点击“暂停”可发信号令RenderThread暂停循环（例如等待condition variable），再点“恢复”继续。
   - **修改渲染参数**：如果用户改变了会影响渲染结果的参数（如采样上限、光线反弹开关等），此时需要通知渲染线程重新开始。典型做法是：设置 RenderThread 内一个标志 `restartRequested=true`，然后立即返回主线程。例如可以在控制面板的槽里调用 RenderThread 提供的 `requestRestart(params)` 方法，内部安全地将标志置位并存储新参数。RenderThread 每次迭代循环顶部检查该标志，若为 true，则中断当前循环：清空累积结果，按新参数重新初始化（或者彻底退出run，由MainWindow重启一个新线程）。随后UI上的显示也可以清空或标记为正在重启。
   - 在重启过程中，可以先让旧线程安全退出，再spawn新线程；或者复用线程对象但重置场景和buffer。考虑实现简单性，**可以选择重启线程**：MainWindow收到参数改动信号后，如果RenderThread正在跑，则调用其 `quit()` 停止，然后 join等待结束，接着以新参数创建并启动一个新的 RenderThread。由于UI纹理是同一个，可以继续使用，只需在新线程开始时清零原HDR累积。这种方案下逻辑清晰，只是重启会有极短暂的停顿。当然也可以更进一步优化为无缝切换，但那更复杂。
5. **结束渲染**：用户点击“停止”或关闭窗口。此时MainWindow会请求 RenderThread 停止（类似上述重启过程但不重启新的）。RenderThread 结束后发出 `finished()` 信号，UI收到后更新界面状态（比如将“停止”按钮变回“开始”可点击等）。程序退出时要确保 RenderThread 正常结束，清理 CUDA 资源（如取消注册 OpenGL资源、释放CUDA内存等）以避免内存泄漏。

### 扩展与备注

- **线程同步**：为保证共享数据正确，可以在 RenderThread 写入纹理/PBO 前，用 `glFinish()` 或 `cudaStreamSynchronize()` 等确保上一次OpenGL使用已完成，防止竞争。另外在 RenderThread map资源前，也应确保主线程不在使用该资源绘制，一般通过信号机制和单缓冲可以满足时序：即每次都是 渲染->通知->绘制->渲染... 循环，不会同时读写。同样，为保险可以用互斥锁保护对HDR帧缓冲的访问，但合理的时序控制应能避免需要显式锁。
- **分辨率改变**：如果用户可以修改渲染分辨率或窗口大小，应处理纹理和缓冲重建。这通常也需要重启渲染线程并重新创建适合新尺寸的OpenGL纹理/PBO等。
- **错误处理**：CUDA初始化或运行错误应及时捕获，通知UI显示错误信息并安全停止线程。Qt 信号槽可以用 `Qt::DirectConnection` 在同线程调用，或 `QueuedConnection` 跨线程。需要注意在应用退出时正确回收线程，否则可能导致程序卡死在关闭阶段。
- **Qt Quick 场景的替代**：如果采用 Qt Quick/QML，那么架构上仍类似，只是 UI 线程变为 QML 的GUI线程。OpenGL显示可以通过 **QQuickFramebufferObject** 或 **QSGTextureProvider** 实现，将CUDA结果作为纹理提供给QML中的Item显示。控制面板则用QML Slider等，通过 signal/slot or Q_PROPERTY 调用后端C++对象。由于本项目选择Qt Widgets，上述就不详述。

综上所述，该架构充分将渲染计算与UI解耦，利用CUDA和OpenGL互操作实现高性能像素传递，并允许用户实时调整显示参数而不中断渲染过程。借助Qt的信号槽，我们实现了后台线程和前台UI的异步协作，在保证界面流畅响应的同时最大化GPU性能利用，满足了题目的所有要求。