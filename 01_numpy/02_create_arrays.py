import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype=np.float64)
print(a.dtype)
#全部为0的矩阵
b = np.zeros((3,4),dtype=np.int16)
print(b)
#全部为1的矩阵
c = np.ones((3,4),dtype=np.int16)
print(c)
#空矩阵
d = np.empty((3,4))
print(d)
#数列
e = np.arange(10,20,2)
print(e)
#重新定义长和宽
f = np.arange(12).reshape(3,4)
print(f)
#生成线段
e = np.linspace(1,10,6,dtype=np.int16).reshape((2,3))
print(e)