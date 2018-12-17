import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

x = np.random.rand(100)
noise = np.asarray(random.sample((range(200)),200))
y =5*x*x+0.1*np.random.randn(100)
poly3 = PolynomialFeatures(degree=3)
X = poly3.fit_transform(x[:,np.newaxis])
clf3 = LinearRegression()
clf3.fit(X,y)

Xplot=poly3.fit_transform(x[:,np.newaxis])
poly3_plot=plt.plot(sorted(x), sorted(clf3.predict(Xplot)), 'r-', label='Cubic Fit')
plt.scatter(x, y, label='Data', color='orange', linestyle='dotted', s=15)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.savefig('Cubicfit.png', facecolor='w', edgecolor='w', pad_inches=0.1,)
plt.show()

