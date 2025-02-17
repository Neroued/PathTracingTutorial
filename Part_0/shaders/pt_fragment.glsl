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