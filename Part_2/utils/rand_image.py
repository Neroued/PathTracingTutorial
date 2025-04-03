import numpy as np
from PIL import Image

# 图像大小
width, height = 512, 512
gamma = 2.2

# 生成 RGB 通道的线性空间随机值 [0.0, 1.0)
rgb_linear = np.random.rand(height, width, 3)

# Gamma 矫正：linear -> display
rgb_gamma = np.power(rgb_linear, 1.0 / gamma)

# 缩放到 [0,255]
rgb_255 = (rgb_gamma * 255).astype(np.uint8)

# Alpha 通道设为 255（不透明）
alpha = np.full((height, width, 1), 255, dtype=np.uint8)

# 拼接 RGBA
rgba = np.concatenate((rgb_255, alpha), axis=2)

# 保存为 PNG 图像
image = Image.fromarray(rgba, mode='RGBA')
image.save("random_rgba_gamma_corrected.png")

print("✅ 已保存 gamma 矫正图像 random_rgba_gamma_corrected.png")
