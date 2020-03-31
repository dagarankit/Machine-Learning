# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:59:55 2020

@author: ankit dagar
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
boston=load_boston()
#print(boston)
'''boston['feature_names']'''
'''['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']'''

'''boston['data'].shape #(506,13)'''

bost = pd.DataFrame(boston['data'])

'''bost.head'''

bost.columns=boston['feature_names']

'''bost.head'''

'''boston['target']'''

''' Normalizing input matrix so that data lies in range of -1 to 1 '''
X=(bost-bost.mean())/(bost.max()-bost.min())  # for better gradient descent

'''X.describe()'''

'''X.head()'''

y=boston['target']

'''y.shape()'''

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=5)

'''x_train.shape() # 339,13  where 339 is no of training examples , 13 no of feautres'''


''' no of training examples to be accross columns where model needs x_training of dimension(no of feature x no of training examples)'''

x_traint = x_train.T

'''x_train.shape()  #13,339'''

'''y_train.shape() # 339 only 1 dimensional'''

''' we need y_train of dimension l x m_train , where m_train = number of training examples'''

y_traint = np.array([y_train])

'''y_train.shape()  # 1,339'''

x_testt = x_test.T

'''x_test.shape()'''

y_test=np.array([y_test])

'''y_test.shape()  # 1,167'''

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_traint,y_traint)

y_pred=model.predict(x_testt)

'''#y_pred.shape()'''

print(mae_val_sklearn=(1/y_test.shape[1])*np.sum(np.abs(y_pred-y_test.T)))

#from sklearn.metrics 











