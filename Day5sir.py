# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:23:20 2020

@author: Saurabh
"""

#ml lecture 5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv(r"E:\Cetpa\Abinash sir ml\Regression.csv")
data.head()

#######################################
#missing value treatment
data.isnull().sum()
data['Age'].fillna(data["Age"].median(),inplace=True)
data['Age'].median()
data.isnull().sum()
######################################################
data.info()
data['Age'].mean()
#######################################
jt=pd.get_dummies(data["Job Type"],drop_first=True)
jt.head()
##########################################
ms=pd.get_dummies(data["Marital Status"],drop_first=True)
ms.head()
############################################
edu=pd.get_dummies(data["Education"],drop_first=True)
edu.head()
########################################
mc=pd.get_dummies(data["Metro City"])
mc.pop("Yes")
mc.head()
##################################################
pre_final=pd.concat((data,jt,ms,edu,mc),axis=1)
pre_final.shape

final_data=pre_final.drop(["Metro City","Education","Marital Status","Job Type"],axis=1)
final_data.shape
#########################################
final_data.isnull().sum()
final_data.to_csv(r"E:\Cetpa\Abinash sir ml\Regression_clean.csv")
############################################
#outlier box plot:- seaborn libarary
import seaborn as sns
sns.boxplot(final_data["Age"])

final_data["Age"]=np.where(final_data["Age"]>58,58,final_data["Age"])
sns.boxplot(final_data["Age"])
#################################################
#sep dependent and independent varaibale
#dep:-purchase made
y=final_data["Purchase made"]
final_data.pop("Purchase made")
x=final_data







