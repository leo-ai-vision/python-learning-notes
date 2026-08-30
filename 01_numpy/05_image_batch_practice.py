# 1. 生成100张32×32的RGB模拟图像
# 2. 数组形状为(100, 32, 32, 3)
# 3. 把像素归一化到0～1
# 4. 计算每个颜色通道的平均值
# 5. 转换成(100, 3, 32, 32)
# 6. 取出前6张图片
# 7. 使用Matplotlib按2×3子图显示
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. 生成100张32×32的RGB图片
images = np.random.randint(
    0,
    256,
    (100, 32, 32, 3),
    dtype=np.uint8,
)

print("原始形状：", images.shape)

# 2. 归一化
images_normalized = images.astype(np.float32) / 255.0

# 3. 计算每个通道的平均值
channel_mean = images_normalized.mean(axis=(0, 1, 2))
print("RGB平均值：", channel_mean)

# 4. NHWC转换为NCHW
images_nchw = images_normalized.transpose(0, 3, 1, 2)
print("转换后形状：", images_nchw.shape)

# 5. 取前6张
first_six = images_nchw[:6]

# 6. 创建2×3子图
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
axes = axes.flatten()
# [
#     [子图1, 子图2, 子图3],
#     [子图4, 子图5, 子图6]
# ]
# [子图1, 子图2, 子图3, 子图4, 子图5, 子图6]

for i, ax in enumerate(axes):
    # 7. CHW转换回HWC
    display_image = first_six[i].transpose(1, 2, 0)

    ax.imshow(display_image)
    ax.set_title(f"Image {i + 1}")
    ax.axis("off")

plt.tight_layout()
plt.show()
