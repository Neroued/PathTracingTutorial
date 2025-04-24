#version 450 core

in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D imageTexture;

uniform float gamma    = 2.2;
uniform float exposure = -2.8;
const float SRGB_ALPHA = 0.055;

// https://github.com/tobspr/GLSL-Color-Spaces
// Converts a single linear channel to srgb
float linear_to_srgb(float channel) {
    if(channel <= 0.0031308)
        return 12.92 * channel;
    else
        return (1.0 + SRGB_ALPHA) * pow(channel, 1.0/2.4) - SRGB_ALPHA;
}

// Converts a linear rgb color to a srgb color (exact, not approximated)
vec3 rgb_to_srgb(vec3 rgb) {
    return vec3(
        linear_to_srgb(rgb.r),
        linear_to_srgb(rgb.g),
        linear_to_srgb(rgb.b)
    );
}

void main() {
    vec3 texCol = texture(imageTexture, TexCoord).rgb;

    // === Tone Mapping ===
    float exposureScale = pow(2.0, exposure);
    texCol              = vec3(1.0) - exp(-texCol * exposureScale);

    // // gamma矫正
    // float invGamma = 1.0 / gamma;
    // vec3 colorGamma;
    // colorGamma.r = pow(texCol.r, invGamma);
    // colorGamma.g = pow(texCol.g, invGamma);
    // colorGamma.z = pow(texCol.z, invGamma);

    // 将线性空间转换为 srgb 
    FragColor = vec4(rgb_to_srgb(texCol), 1.0);
}