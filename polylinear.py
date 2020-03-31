# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:26:55 2020

@author: ankit dagar
"""

''' polynomial regression'''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

x=2-3 * np.random.normal(0,1,20)
y=x-2 * (x**2) + 0.5 * (x**3) + np.random.normal(-3,3,20)
plt.scatter(x,y)
plt.show()

from sklearn.linear_model import LinearRegression
x=x[:,np.newaxis]
y=y[:,np.newaxis]

model = LinearRegression()
model.fit(x,y)
y_pred=model.predict(x)

#plt.scatter(x,y)
#plt.plot(x,y_pred,color='red')
#plt.show()

from sklearn.metrics import r2_score,mean_squared_error
r2=r2_score(y,y_pred)
rmse=(np.sqrt(mean_squared_error(y,y_pred)))
print(r2) #'''0.6386750054827146'''
print(rmse) #'''15.908242501429998'''

'''to overcome underfitting we  need to increase the complexity of model'''

#import operator
from sklearn.preprocessing import PolynomialFeatures

polynomial_features = PolynomialFeatures(degree=2)
x_poly=polynomial_features.fit_transform(x)

modelnew=LinearRegression()
modelnew.fit(x_poly,y)
y_poly_pred = modelnew.predict(x_poly)

rmsenew = np.sqrt(mean_squared_error(y,y_poly_pred))
r2new = r2_score(y,y_poly_pred)

print(rmsenew)
print(r2new)

#10.120437473614711
#0.8537647164420812

plt.scatter(x,y)
plt.show()



'ggplot'