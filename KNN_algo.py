# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:34:47 2020

@author: ankit dagar
"""
#MODEL SELECTION,TUNING,EALUATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

dataset = load_iris()
X=dataset.data     #to show data where it is
y=dataset.target   #to show target  continous regression categorical classification

plt.scatter(X[y == 0, 0],X[y== 0, 1], c='r', label='Setosa')
plt.scatter(X[y == 1, 0],X[y== 1, 1], c='g', label='Versicolor')
plt.scatter(X[y == 2, 0],X[y== 2, 1], c='b', label='Virginica')
plt.xlabel('Sepal Length')
plt.legend()
plt.ylabel('Sepal Width')
plt.title('Analysis on the Iris Dataset')
plt.show()
plt.scatter(X[y == 0, 2],X[y== 0, 3], c='r', label='Setosa')
plt.scatter(X[y == 1, 2],X[y== 1, 3], c='g', label='Versicolor')
plt.scatter(X[y == 2, 2],X[y== 2, 3], c='b', label='Virginica')
plt.xlabel('Petal Length')
plt.legend()
plt.ylabel('Petal Width')
plt.title('Analysis on the Iris Dataset')
plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)     
knn.fit(X,y)
knn.score(X,y) ## accuracy ((tn+tp)/m)
print(knn.score(X,y))
y_pred=knn.predict(X)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y,y_pred)
print(classification_report(y,y_pred))
#from sklearn.metrics import precision_score, recall_score
#precision_score(y_pred)
#print(precision_score)
#recall_score()

# for k=3 model accuracy is 96







