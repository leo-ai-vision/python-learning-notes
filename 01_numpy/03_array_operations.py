import numpy as np
# a = np.array([10,20,30,40])
# b = np.arange(4)
# c = a - b
# d = a * b
# e = b ** 2
# f = 10 * np.sin(a)
# print(f)
# print(b<3)
#矩阵的运算
# a = np.array([[1,1],[0,1]])
# b = np.arange(4).reshape((2,2))
# c = a*b             #逐个相乘
# c_dot = np.dot(a,b) #矩阵乘法
# c_dot_2 = a.dot(b)
# print(c_dot)
# print(c_dot_2)
# print(c)

#随机生成2行4列矩阵
# d = np.random.random((2,4))
# print(d)
# print(np.sum(d,axis=1))#axis=1 行    axis=0 列
# print(np.min(d,axis=0))
# print(np.max(d))

A = np.arange(2,14).reshape(3,4)
print(np.argmin(A))
print(np.argmax(A))
print(np.mean(A))#平均值
print(A.mean())
print(np.average(A))#平均值
print(np.median(A))#中位数
print(np.cumsum(A))# 逐项累加
print(np.diff(A))#相邻两数的差  1，2   2，3   3，4
print(np.nonzero(A))
print(A)
B= np.arange(14,2,-1).reshape(3,4)
print(B)
print(np.sort(B))#每一行升序排序
print(np.transpose(B))#转置
print(B.T)#转置
print(np.clip(B,5,9))#裁剪
