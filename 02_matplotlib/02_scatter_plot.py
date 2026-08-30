import numpy as np
import matplotlib.pyplot as plt

n= 1024#数据个数 x轴1024，y轴1024
X = np.random.normal(0,1,n) #平均数0 方差1 个数n
Y = np.random.normal(0,1,n)
T = np.arctan2(Y,X)#为了颜色
#              大小75颜色T 透明度
plt.scatter(X,Y,s=75,c=T,alpha=0.5)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.xticks(())  # 隐藏 x 轴刻度和刻度标签
plt.yticks(())  # 隐藏 y 轴刻度和刻度标签


plt.show()