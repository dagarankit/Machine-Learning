# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 20:11:17 2020

@author: ankit dagar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

dataset=load_iris()
X=dataset.data
y=dataset.target

plt.scatter(X[y==0,0],X[y==0,1],c='r',label='setosa')
plt.scatter(X[y==1,0],X[y==1,1],c='g',label='versiclor')
plt.scatter(X[y==2,0],X[y==2,2],c='b',label='virginica')
plt.xlabel('Sepal Length')
plt.legend()
plt.ylabel('Sepal Width')
plt.title('Analysis on the iris dataset')
plt.show()

