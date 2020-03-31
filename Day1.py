# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:16:33 2020

@author: Saurabh

"""
#lecture 1 ML

import numpy as np
a=np.array([1,2,3,4])
print(a)
print(type(a))


#nested list
l=[[1,2,3],[4,5,6],[7,8,9]]
a=np.array(l)
print(a)

#dimension

print(a.ndim)

#shape
print(a.shape)


#dtype
print(a.dtype)


#1D list
a=np.array([1,5,6])
a=np.array(a)
print(a)
#dimension

print(a.ndim)

#shape
print(a.shape)


#dtype
print(a.dtype)

#size
print(a.size)


#Array creation
a=np.array([1,2,3,4,5],dtype="float")
print(a)

#zeroes and ones
#used in sending perceptron(like neuorn in human) in Deep learning
a=np.zeros((3,4))
print(a)

a=np.ones((3,4))
print(a)


#Random numpy array
#The above method creates a matrix of 2*2 whose random value lies b/w 0 to 1. of float type.
n=np.random.random((2,2))
print(n)

#arange function
a=np.arange(0,100,5)
print(a)
print(type(a))


l=[[1,5,6,7],[5,8,9,9],[15,10,20,15]]
a=np.array(l)
arr=a.reshape((6,2))
print(arr)
arr1=a.reshape(2,2,3)
print(arr1)

arr=a.reshape((5,2))


#Flatten method
a=np.array([[1,2,3],[3,5,6]])
d=a.flatten()
print(d)


#Array indexing and Slicing
a=np.array([[1,2,5],[25,6,7],[27,7,25]])
print(a[0:1,0:1])

a=np.array([[1,2,5,7,8],[25,6,7,6,7],[27,7,25,22,43]])
print(a[0:1,0:1])


#all row and index per element
print(a[0:,0::+2])

print(a[0:,::-1])


#mathmatical operations
a=[5,5,5]
b=[1,5,2]
print(a/b)
#this opeartion cant perform in list
#to overcome this we use numpy array

a=np.array(a)
b=np.array(b)
print(a/b)
#Square and all operations
a=[1,4,9,16,25]
a=np.array(a)
print(a**2)
print(a+2)
print(a-2)
print(a*1.414)


#transpose the matrix
a=[[1,2],[3,4],[5,6]]
arr=np.array(a)
arr_t=arr.T
print(arr_t)

#binary opeartions
#operations is according to the index
a=[[1,2],[3,4],[5,6]]
a_a=np.array(a)
b=[[2,3],[4,5],[6,7]]
b_b=np.array(b)
print(a_a+b_b)

#
#neseteed list
l=[[1,5,6,7],
   [5,8,9,9],
   [15,10,20,15]]
#ndarray
arr=np.array(l)
print(arr.sum())
print(arr.min())
print(arr.max())


print(arr.max(axis=1))#row wise max
print(arr.max(axis=0))#col wise max


print(arr.min(axis=1))#row wise min
print(arr.min(axis=0))#col wise min

#sorting
print(np.sort(arr))

print(np.sort(arr,axis=1))#row wise sort
print(np.sort(arr,axis=0))#col wise sort


d=[[1,5,6,7],
   [5,8,9,9],
   [15,10,20,15]]

print(d[a:,::-1])


#stacking
#Verical and horizontal stack
arr1=np.array([[1,2],[3,5]])
arr2=np.array([[5,6],[18,20]])
print(np.vstack((arr1,arr2)))
print(np.hstack((arr1,arr2)))










