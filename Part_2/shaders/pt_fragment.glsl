#version 450 core

in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D imageTexture;

void main()
{
    vec3 texCol = texture(imageTexture, TexCoord).rgb;

    // === Tone Mapping（可选：Reinhard / Exponential）===
    // 方案一：Exponential（柔和一些）
    // 曝光度 0.5
    texCol = vec3(1.0) - exp(-texCol * 0.5);

    // gamma矫正
    float invGamma = 1.0 / 2.2;

    vec3 colorGamma;
    colorGamma.r = pow(texCol.r, invGamma);
    colorGamma.g = pow(texCol.g, invGamma);
    colorGamma.z = pow(texCol.z, invGamma);

    FragColor = vec4(colorGamma, 1.0);
}