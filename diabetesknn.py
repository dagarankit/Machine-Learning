# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:29:33 2020

@author: ankit dagar
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
dataset=load_diabetes()
X=dataset.data
y=dataset.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
knn.score(x_train,y_train)

print(knn.score(x_train,y_train))


from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))