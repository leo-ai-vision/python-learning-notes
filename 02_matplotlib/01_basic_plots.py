import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,50)
y1 = 2*x+1
y2 = x**2
plt.figure()
plt.plot(x,y1)
#          命名figure3，长8，宽5
plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
#                                 线宽1.0       虚线
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
plt.show()
