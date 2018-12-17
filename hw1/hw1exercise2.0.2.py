
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression



x = np.random.rand(100)
y = 5*x*x+0.1*np.random.randn(100)
poly3 = PolynomialFeatures(degree=3)
X = poly3.fit_transform(x[:,np.newaxis])
clf3=LinearRegression
clf3.fit(X,y)

Xplot = poly3.fit_transform(x[:,np.newaxis])
poly3_plot = plt.plot(x, clf3.predict(Xplot),label='Cubic Fit')
plt.scatter(x, y, label='Data', color='orange', s=15)
plt.legend()
plt.savefig('Cubic fit', facecolor='w', edgecolor='w', pad_inches=0.1,)
plt.show()

def error(a):
    for i in y:
        err=(y-yn)/yn
    return abs(np.sum(err))/len(err)

print (error(y))

'''
linreg=LinearRegression()
linreg.fit(x,y)
xnew = np.array([[0],[1]])
ypredict = linreg.predict(xnew)
'''

'''
#plotting the data.
plt.plot(xnew, ypredict, "r-")
plt.plot(x, y ,'ro')
plt.axis([0,1.0,0, 5.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Simple Linear Regression')
plt.show()
'''


'''
#plotting the relative error
plt.plot(x, np.abs(ypredict-y)/abs(y), "ro")
plt.axis([0,1.0,0.0, 0.5])
plt.xlabel(r'$x$')
plt.ylabel(r'$\epsilon_{\mathrm{relative}}$')
plt.title(r'Relative error')
plt.savefig('RelativeError', facecolor='w', edgecolor='w', pad_inches=0.1,)
plt.show()
'''