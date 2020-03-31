# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:22:11 2020

@author: ankit dagar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

dataset=load_breast_cancer()
#print(dataset.data)
X=dataset.data
y=dataset.target

plt.scatter(X[y==0,0],X[y==0,1],c='r')
plt.scatter(X[y==1,0],X[y==1,1],c='g')
plt.xlabel('Id')
plt.legend()
plt.ylabel('Diagnosis')
plt.show()
plt.title('Breast Cancer')
plt.scatter(X[y==0,2],X[y==0,3],c='r')
plt.scatter(X[y==1,2],X[y==1,3],c='g')
plt.xlabel('Malignant')
plt.legend()
plt.ylabel('Benign')
plt.title('Diagnosis Type')
plt.show()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30)



from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn.score(x_train,y_train)
print(knn.score(x_train,y_train))


y_pred=knn.predict(x_test)
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))

 