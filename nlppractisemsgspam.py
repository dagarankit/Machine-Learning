import nltk
import pandas as pd
message=pd.read_csv('C:/Users/Aaadityaa/Downloads/Documents/SMSSpamCollection',sep='\t',names=['label','message'])
message.describe()
message.groupby('label').describe()
message['length']=message['message'].apply(len)

#VISUALIZATION OF DATA
import matplotlib.pyplot as plt
import seaborn as sns

message['length'].plot(bins=60,kind='hist')
message.length.describe()

message[message['length']== 910]['message'].iloc[0]


#REMOVE PUNCATUATION

import string
mess= 'sample message! Notice: it has punctuation.'

nopunc=[char for char in mess if char not in string.punctuation]

nopunc=''.join(nopunc)

import nltk
from nltk.corpus import stopwords
stopwords.words('english')[0:10]

nopunc.split()

#now just remove any stopword
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





message_bow=bow_transformer.transform(message['message'])
print('shape of sparse matrix:',message_bow.shape)
print('Amount of Non-Zero occurnces: ',message_bow.nnz)

from sklearn.feature_extraction.text



from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer.transform(message_bow)

from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test=train_test_split(mssage['label'],test_size=0.2)
print(len(msg_train))