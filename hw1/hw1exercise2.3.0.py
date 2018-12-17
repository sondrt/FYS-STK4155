import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error


x = np.random.rand(100)*2
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
#plt.show()

linreg = LinearRegression()
linreg.fit(X,y)
ypredict = linreg.predict(X)

# Mean squared log error                                                        
print('Mean squared log error: %.2f' % mean_squared_log_error(y, ypredict) )

# Mean absolute error                                                           
print('Mean absolute error: %.2f' % mean_absolute_error(y, ypredict))



'''


#print('The intercept alpha: \n', linreg.intercept_)
print('Coefficient beta : \n', linreg.coef_)

# The mean squared error                               
print("Mean squared error: %.2f" % mean_squared_error(y, ypredict))

# Explained variance score: 1 is perfect prediction                                 
print('Variance score: %.2f' % r2_score(y, ypredict))

# Mean squared log error                                                        
print('Mean squared log error: %.2f' % mean_squared_log_error(y, ypredict) )

# Mean absolute error                                                           
#print('Mean absolute error: %.2f' % mean_absolute_error(y, ypredict))
plt.plot(x, ypredict, "r-")
plt.plot(x, y ,'ro')
plt.axis([0.0,1.0,1.5, 7.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Linear Regression fit ')
plt.show()

'''
