# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:13:46 2020

@author: ankit dagar
"""
import math

def zscore():
    xb=int(input("Enter value of x bar"))
    stad_dev=float(input("enter value of standard deviation"))
    nosamp=int(input("enter no of samples"))
    u=int(input("enter u"))
    z_score=(xb-u)/((stad_dev)/(math.sqrt(nosamp)))
    print(z_score)
    
zscore()