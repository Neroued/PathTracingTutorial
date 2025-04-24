#version 330 core
layout (location = 0) in vec2 aPos;   // 顶点位置 (来自 VBO)
layout (location = 1) in vec2 aTexCoords; // 纹理坐标 (来自 VBO)

out vec2 TexCoords; // 传递给片段着色器的纹理坐标

uniform mat4 viewMatrix = mat4(1.0); // 视图矩阵 (用于平移和缩放)

void main()
{
    // 直接使用四边形的顶点位置，应用视图变换
    gl_Position = viewMatrix * vec4(aPos.x, aPos.y, 0.0, 1.0);
    // 将纹理坐标传递给片段着色器
    TexCoords = aTexCoords;
}

