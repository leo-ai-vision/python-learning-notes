import numpy as np
A = np.arange(3,15).reshape((3,4))
# print(A)
# print(A[1][1])#1行1列
# print(A[1,1])
# print(A[2,:])
# print(A[1,1:3])
# for row in A:
#     print(row)#  每一行
# for col in A.T:
#     print(col) #每一列
print(A.flatten())#展开为一行
for item in A.flat:
    print(item)
