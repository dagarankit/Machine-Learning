# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:23:06 2020

@author: Saurabh



"""
ML lecture 4 ML



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df1={"name":["a","b","c","d","e","f"],
      "age":[1,2,3,4,5,6],
      "edu":["pg","ug","phd","phd","pg","ug"]}

df2={"name":["a","b","c","d","e","f"],
      "age":[1,2,3,4,5,6]}


d1=pd.DataFrame(df1)
d2=pd.DataFrame(df2)

net_df=pd.merge(d1,d2,on='a')
print(net_df)



##########################################################
import numpy as np
import matplotlib as pl
import matplotlib.pyplot as plt
#matplotlib built upon numpy array
x=[1,2,3,4,5,6]
y=[11,22,33,44,55,66]
plt.plot(x,y)
plt.show()


############################################################
scatter plot


x=[1,2,3,4,5,6]
y=[11,22,33,44,55,66]
plt.scatter(x,y,color="red")
plt.show()


#################################################################
bar plot

x=[1,2,3,4,5,6]
y=[11,22,33,44,55,66]
plt.bar(x,y,color="red")
plt.show()

###############################################################
hist


x=[1,2,3,4,5,6]
y=[11,22,33,44,55,66]
plt.hist(x)
plt.show()

x=[1,2,3,4,5,6]
y=[11,22,33,44,55,66]
plt.hist(y)
plt.show()



############################################################
#naming axis

x=[1,2,3,4,5,6]
y=[11,22,33,44,55,66]
plt.plot(x,y)
plt.legend()
plt.title("mygraph")
plt.xlabel("X-axis")
plt.ylabel("y-axis")
plt.show()

##########################33
import matplotlib.pyplot as plt
#line1
x1=[1,2,3]
y1=[2,4,1]
plt.plot(x1,y1,label="line 1")


#line2
x2=[1,2,3]
y2=[4,1,3]
plt.plot(x2,y2,label="line 2")
plt.title("mygraph")
plt.xlabel("X-axis")
plt.ylabel("y-axis")
plt.legend()
plt.show()
#####################################################
x=[1,2,3,4,5,6]
y=[2,3,1,5,2,6]

plt.plot(x,y,color="black",linestyle='dashed',linewidth=3,marker='o',markerfacecolor='red',markersize=10)

plt.ylim(1,8)
plt.xlim(1,8)
plt.ylabel("y axis")
plt.xlabel('x axis')
plt.title("mygraph")
plt.show()

##############################################



x=[1,2,3,4,5,6]
y=[2,3,1,5,2,6]

plt.plot(x,y,color="black",linewidth=3,marker='o',markerfacecolor='red',markersize=10)

plt.ylim(1,8)
plt.xlim(1,8)
plt.ylabel("y axis")
plt.xlabel('x axis')
plt.title("mygraph")
plt.show()

#######################################################

a=[1,2,3,4,5]
h=[10,24,36,42,100]

t=['one','two','three','four','five']
plt.bar(a,h,tick_label=t,width=0.5,color=['red','green'])

plt.show()
###############################################
ages=[24,32,56,98,45,52,31,42,19,12]
rangee=(0,100)
bins=10

plt.hist(ages,bins,rangee,color="green",histtype='bar',rwidth=0.8)
plt.show()

######################################################################


x=[1,2,3,4,5,6]
y=[2,3,1,5,2,6]

plt.scatter(x,y,label="stars",color="red",marker='*',s=50)
plt.show()

#############################################################
#pie chart


act=['a','b','c','d']
s=[3,8,9,11]
color=['r','y','g','b']


plt.pie(s,labels=act,colors=color,startangle=90,
        explode=(0,0,0.1,0),shadow=True,radius=1.1,
        autopct="%1.1f%%")
plt.legend()
plt.show()

###############################################################
shadow=True show a shadow beneath each label in pie chart
































