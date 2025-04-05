#version 450 core

in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D imageTexture;

uniform float gamma    = 2.2;
uniform float exposure = -2.8;

void main() {
    vec3 texCol = texture(imageTexture, TexCoord).rgb;

    // === Tone Mapping ===
    float exposureScale = pow(2.0, exposure);
    texCol              = vec3(1.0) - exp(-texCol * exposureScale);

    // gamma矫正
    float invGamma = 1.0 / gamma;
    vec3 colorGamma;
    colorGamma.r = pow(texCol.r, invGamma);
    colorGamma.g = pow(texCol.g, invGamma);
    colorGamma.z = pow(texCol.z, invGamma);

    FragColor = vec4(colorGamma, 1.0);
}