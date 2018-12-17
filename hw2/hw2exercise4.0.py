import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#creating data
x = np.random.rand(100)
y =5*x*x+0.1*np.random.randn(100)

# arranging data into 2x100 matrixes
a = np.array(x)
b = np.array(y)

#split into tranining and test data
X_train = a[:50,np.newaxis]
X_test = a[50:,np.newaxis]
y_train = b[:50]
y_test = b[50:]

print ("X_train: ", X_train.shape)
print ("y_train: ", y_train.shape)
print ("X_test: ", X_test.shape)
print ("y_test: ", y_test.shape)
