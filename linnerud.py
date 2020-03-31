# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:23:34 2020

@author: ankit dagar
"""

from sklearn.datasets import load_linnerud
dataset=load_linnerud()
X=dataset.data
y=dataset.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,x_test)
y_pred=knn.predict(x_test)
knn.score(x_train,y_train)

print(knn.score(x_train,y_train))

#from sklearn.metrics import confusion_matrix,classification_report
#cm=confusion_matrix(y_test,y_pred)
#print(classification_report(y_test,y_pred))

