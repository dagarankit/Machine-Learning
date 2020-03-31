# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 13:15:32 2020

@author: ankit dagar
"""
import nltk
import pandas as pd

message=pd.read('F:\ML\LR\SMSSpamCollection',sep='\t',names=['label','message'])

message.describe()

message.groupby('label').describe()

message['length']=message['message'].apply(len)

import matplotlib.pyplot as plt
import seaborn as sns

message['length'].plot(bins=60,kind='hist')

message.length.describe()

message[message['length']==910]['message'].iloc[0]

import string

mess = 'sample message! Notice: it has punctuation.'

nopunc=[char for char in mess if char not in string.punctuation]

nopunc=''.join(nopunc)

import nltk

from nltk.corpus import stopwords

stopwords.words('english')[0:10]

nopunc.split()


clean_mess=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
def text_process(mess):
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)    
    return[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
 message['mess']=message['message'].apply(text_process)


from sklearn.feature_extraction.text import CountVectorizer
bow_transformer=CountVectorizer(analyzer=text_process).fit(message['message'])
print(len(bow_transformer.vocabulary_))

message4=message['message'][3]
