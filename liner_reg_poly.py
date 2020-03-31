# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:38:39 2020

@author: ankit dagar
"""
#import operator 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
x=2-3 * np.random.normal(0,1,20)
y=x-2*(x**2) + 0.5 *(x**3)+np.random.normal(-3,3,20)

x=x[:,np.newaxis]
y=y[:,np.newaxis]

polynomial_features=PolynomialFeatures(degree=2)  #for manipulating features we can take degree
x_poly=polynomial_features.fit_transform(x)

model=LinearRegression()
model.fit(x_poly,y)
y_poly_pred=model.predict(x_poly)
#model.score(x,y)

mse=np.sqrt(mean_squared_error(y,y_poly_pred))

rscore=r2_score(y,y_poly_pred)
print(mse)
print(rscore)

plt.scatter(x,y)
plt.show()




#rmsenp