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