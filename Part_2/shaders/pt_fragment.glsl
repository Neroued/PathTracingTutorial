#version 450 core

in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D imageTexture;

void main()
{
    vec3 texCol = texture(imageTexture, TexCoord).rgb;

    // gamma矫正
    float invGamma = 1.0 / 2.2;

    vec3 colorGamma;
    colorGamma.r = pow(texCol.r, invGamma);
    colorGamma.g = pow(texCol.g, invGamma);
    colorGamma.z = pow(texCol.z, invGamma);

    FragColor = vec4(colorGamma, 1.0);
}