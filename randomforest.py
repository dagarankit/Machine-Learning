# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:53:55 2020

@author: ankit dagar
"""

from sklearn.datasets import load_wine
loadwine=load_wine()

X=loadwine.data
y=loadwine.target

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=77)

from sklearn.preprocessing import MinMaxScaler

skl=MinMaxScaler()
skl.fit(X_train)

x_train_scaled=skl.transform(X_train)

print("transformedshape: %s" % (x_train_scaled.shape,))
print("per-feature minimum before scaling:\n %s" % X_train.min(axis = 0))
print("per-feature maximum before scaling:\n %s" % X_train.max(axis = 0))
print("per-feature minimum after scaling:\n %s" % x_train_scaled.min(axis = 0))
print("per-feature maximum after scaling:\n %s" % x_train_scaled.max(axis = 0))

x_test_scaled=skl.transform(X_test)

print("per-feature min after scaling: %s" % x_test_scaled.min(axis=0))
print("per-feature max after scaling: %s" % x_test_scaled.max(axis=0))


from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)

model.fit(X,y)
model.fit(X,y)

model.score(X_train,y_train)
print(model.score(X_train,y_train))

model.score(X_test,y_test)

print(model.score(X_test,y_test))

from sklearn.metrics import classification_report,confusion_matrix

y_pred=model.predict(X_test)

cl=classification_report(y_test,y_pred)

cl

cm=confusion_matrix(y_test,y_pred)

