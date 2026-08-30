import matplotlib.pyplot as plt
import numpy as np

n =12#12个柱状图
X = np.arange(n)
# 生成第一组柱子的高度：
# (1 - X/n) 使柱高随着 X 增大而逐渐减小
# np.random.uniform(0.5, 1.0, n) 生成 n 个 0.5～1.0 之间的随机数
# uniform随机生成 n 个数，每个数都在 0.5 到 1.0 之间，并且区间内各个位置被选中的机会相同
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

# 生成第二组柱子的高度，计算方式与 Y1 相同，
# 但会重新生成一组随机数，所以 Y2 和 Y1 通常不相同
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

plt.xlim(-5,n)
plt.xticks(())
plt.ylim(-1.25,1.25)
plt.yticks(())
plt.bar(X,+Y1,facecolor='#9999ff',edgecolor='white')
plt.bar(X,-Y2,facecolor='#ff9999',edgecolor='white')
# zip(X, Y1) 将 X 和 Y1 中相同位置的元素配成一对
# 循环取得每根上方柱子的横坐标 x 和高度 y
for x, y in zip(X, Y1):
    # 在柱子上方显示数值
    # x + 0.4：文字横坐标，0.4 用于移动到柱子中心
    # y + 0.05：文字纵坐标，放在柱顶稍微偏上的位置
    # '%.2f' % y：把 y 格式化为保留两位小数
    # ha='center'：文字水平居中
    # va='bottom'：文字底部与指定坐标对齐，文字向上显示
    plt.text(
        x + 0.4,
        y + 0.05,
        '%.2f' % y,
        ha='center',
        va='bottom'
    )


# 循环取得每根下方柱子的横坐标 x 和高度 y
for x, y in zip(X, Y2):
    # 在向下的柱子下方显示负数数值
    # x + 0.4：移动到柱子中心
    # -y - 0.05：柱子向下，所以坐标使用负数，并再向下偏移 0.05
    # '-%.2f' % y：显示负号，并保留两位小数
    # ha='center'：文字水平居中
    # va='top'：文字顶部与指定坐标对齐，文字向下显示
    plt.text(
        x + 0.4,
        -y - 0.05,
        '-%.2f' % y,
        ha='center',
        va='top'
    )

plt.show()
