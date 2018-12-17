# Simple regression model.

import numpy as np 

#print(np.c_[np.array([1,2,3]), np.array([4,5,6])])
#print(np.c_[np.array([[1,2,3]]), 0,0,np.array([[4,5,6]])])
#print (np.c_[np.array([[1,2,3]]),0,0])

from random import random, seed
import matplotlib.pyplot as plt 

x = 2*np.random.rand(100,1)
y = 4+ 3*x + np.random.randn(100,1)*

xb = np.c_[np.ones((100,1)),x]
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
xnew = np.array([[0],[2]])
xbnew = np.c_[np.ones((2,1)),xnew]
ypredict = xbnew.dot(beta)

plt.plot(xnew, ypredict, "r-")
plt.plot(x,y,"ro")
plt.axis([0,2.0,0,15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'linaer Regression')
plt.show()