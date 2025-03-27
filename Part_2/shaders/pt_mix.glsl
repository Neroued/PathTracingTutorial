#version 450 core

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(rgba32f, binding = 0) uniform image2D computeTexture; // 输入的单次采样结果
layout(rgba32f, binding = 1) uniform image2D imageTexture;  // 输出的混合后的图像

layout(location = 0) uniform uint sampleCount;

void main()
{
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);

    vec4 newSample = imageLoad(computeTexture, texelCoord);
    vec4 accSample = imageLoad(imageTexture, texelCoord);

    accSample = (accSample * (sampleCount - 1) + newSample) / sampleCount;
    imageStore(imageTexture, texelCoord, accSample);
}