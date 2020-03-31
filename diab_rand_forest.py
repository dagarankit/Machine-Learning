# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:03:23 2020

@author: ankit dagar
"""

from sklearn.datasets import load_diabetes
diab=load_diabetes()

X=diab.data
y=diab.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler.fit(X_train)

x_train_scalednew=scaler.transform(X_train)
y_train_scalednew=scaler.transform(y_train)

print("transsformed shape: %s" % (x_train_scalednew,))
print("per feature min before scaling: %s" % X_train.min(axis=0))
print("per feature max before scaling: %s" % X_train.max(axis=0))
print("per feature min after scaling: %s" % x_train_scalednew.min(axis=0))
print("per feature max after scaling: %s" % x_train_scalednew.max(axis=0))


x_test_scalednew=scaler.transform(X_test)
y_test_scalednew=scaler.transform(y_test)

print("per-feature min after scaling: %s" % x_test_scalednew.min(axis=0))
print("per-feature max after scaling: %s" % x_test_scalednew.max(axis=0))

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50)

model.fit(X,y)

acc_train=model.score(X_train,y_train)
acc_test=model.score(X_test,y_test)

y_pred=model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

cl=classification_report(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)






