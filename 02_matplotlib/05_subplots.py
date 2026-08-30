
import matplotlib.pyplot as plt
# example 1:
###############################
plt.figure()
# 将当前画布划分为 2 行 2 列，共 4 个子图
# 最后的 1 表示选择第 1 个子图，也就是左上角
plt.subplot(2, 2, 1)

# 在当前选择的第 1 个子图中绘制一条直线
# 横坐标数据：[0, 1]
# 纵坐标数据：[0, 1]
# 连接点 (0, 0) 和 (1, 1)
plt.plot([0, 1], [0, 1])

plt.subplot(2, 2, 2)
plt.plot([0,1], [0,2])
plt.subplot(2, 2, 3)
plt.plot([0,1], [0,3])
plt.subplot(2, 2, 4)
plt.plot([0,1], [0,4])
# example 2:
###############################
plt.figure()
plt.subplot(2, 1, 1)
plt.plot([0,1], [0,1])
plt.subplot(2, 3, 4)
plt.plot([0,1], [0,2])
plt.subplot(2, 3, 5)
plt.plot([0,1], [0,3])
plt.subplot(2, 3, 6)
plt.plot([0,1], [0,4])

plt.tight_layout()#自动调整图中各个子图之间的间距，尽量避免标题、坐标轴名称和刻度文字相互重叠或超出画布。

plt.show()