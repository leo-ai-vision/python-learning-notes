import matplotlib.pyplot as plt
import numpy as np

# image data
# 创建包含 9 个小数的一维 NumPy 数组
# reshape(3, 3) 将一维数组变成 3 行 3 列的二维数组
a = np.array([
    0.313660827978, 0.365348418405, 0.423733120134,
    0.365348418405, 0.439599930621, 0.525083754405,
    0.423733120134, 0.525083754405, 0.651536351379
]).reshape(3, 3)

# 此时 a 相当于：
# [[0.31366083, 0.36534842, 0.42373312],
#  [0.36534842, 0.43959993, 0.52508375],
#  [0.42373312, 0.52508375, 0.65153635]]


"""
for the value of "interpolation", check this:
http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
for the value of "origin"= ['upper', 'lower'], check this:
http://matplotlib.org/examples/pylab_examples/image_origin.html
"""
# 将二维数组 a 显示为图像
plt.imshow(
    a,

    # 像素插值方法
    # 'nearest' 表示使用最近邻插值，不平滑像素之间的颜色
    # 因此可以清晰看到 3×3 的色块
    interpolation='nearest',

    # 设置颜色映射
    # 'bone' 是一种由黑色、灰色到接近白色的配色方案
    cmap='bone',

    # 设置数组第一行显示的位置
    # 'upper' 表示数组第一行显示在图像顶部
    # 如果设置为 'lower'，第一行会显示在图像底部
    origin='upper'
)

# 添加颜色条，用于说明不同颜色对应的数值
# shrink=0.92 表示把颜色条缩小到默认高度的 92%
plt.colorbar(shrink=0.92)

plt.xticks(())
plt.yticks(())

# 显示图像
plt.show()
