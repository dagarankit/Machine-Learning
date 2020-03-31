# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:08:54 2020

@author: Saurabh
"""

#Ml lecture 2

########PANDAS#########
import pandas as pd


#pandas :1)series(one dimension data),2)dataframe(row column),3)panel data(3d data))



#DATAFRAME
#key will become heading
data={"name":["a","b","c","d","e","f"],
      "age":[1,2,3,4,5,6],
      "edu":["pg","ug","phd","phd","pg","ug"]}

print(type(data))

#convert into dataframe
import pandas as pd
df=pd.DataFrame(data)
print(df)
print(type(df))



###########################################
#extract only name from df

print(df["name"])
print(df["age"]>=4)


a=df["age"]
for i in a:
    if(i>4):
        print(i)
        

#import data from external resource
#read from csv
data=pd.read_csv(r"C:\Users\Saurabh\Desktop\abc.csv")
print(data)


#multiple sheet
#read from excelsheet
data_pm=pd.read_excel(r"C:\Users\Saurabh\Desktop\Book1.xlsx",sheet_name="1")

data_sc=pd.read_excel(r"C:\Users\Saurabh\Desktop\Book1.xlsx",sheet_name="2")

print(data_sc)

###################################################

#big dataset
dataset=pd.read_csv(r"C:\Users\Saurabh\Desktop\Salaries.csv")
################################################
#dataset info:
#moneyball
dataset.info()

#datset describe

dataset.describe()


#################################
#dataframe convert into a csv file
data={"name":["a","b","c","d","e","f"],
      "age":[1,2,3,4,5,6],
      "edu":["pg","ug","phd","phd","pg","ug"]}

print(type(data))

#convert into dataframe
import pandas as pd
df=pd.DataFrame(data)
df.to_csv(r"C:\Users\Saurabh\Desktop\mydata.csv")

################################
#extract rows and columns
#iloc method
#rows
df.iloc[0:5]

#specific rows and specific column
s_c_r=dataset.iloc[0:5,1:7]
print(s_c_r)


s_c_r_n=dataset.iloc[0:,0::+2]
print(s_c_r_n)
##############################
#extracting columns by name
data_aa=dataset.loc[0:,["Id","BasePay","Benefits"]]
print(data_aa)
df_merge=pd.merge(data_sc,df)

print(df_merge)
#hw join
#hw to merge

############################33
#missing vaklue analysis
dataset.isnull()
dataset.isnull().sum()




####################################################3




