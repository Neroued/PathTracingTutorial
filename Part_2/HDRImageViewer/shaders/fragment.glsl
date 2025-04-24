#version 330 core
out vec4 FragColor; // 输出的最终颜色

in vec2 TexCoords; // 从顶点着色器接收的纹理坐标

uniform sampler2D hdrTexture; // HDR 图像纹理 (sampler2D 用于浮点纹理)
uniform float exposure = 1.0; // 曝光控制

// 简单的 Reinhard 色调映射函数
vec3 reinhardToneMapping(vec3 color) {
    color *= exposure; // 应用曝光
    return color / (color + vec3(1.0));
}

// 简单的 ACES Filmic 色调映射 (近似)
// Source: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 acesFilmicToneMapping(vec3 color) {
    color *= exposure; // 应用曝光
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}


void main()
{
    // 从 HDR 纹理中采样颜色值 (已经是 float 类型)
    vec3 hdrColor = texture(hdrTexture, TexCoords).rgb;

    // --- 色调映射 ---
    // 选择一种色调映射算法
    // vec3 mappedColor = reinhardToneMapping(hdrColor);
    vec3 mappedColor = acesFilmicToneMapping(hdrColor); // 推荐使用 ACES

    // --- Gamma 校正 ---
    // 通常显示器需要 Gamma 约为 2.2 的校正
    float gamma = 2.2;
    vec3 gammedColor = pow(mappedColor, vec3(1.0 / gamma));

    // 输出最终颜色 (LDR)
    FragColor = vec4(gammedColor, 1.0);
}

