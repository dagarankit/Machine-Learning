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

#data cleaning and preprocessing
from nltk.stem import WordNetLemmatizer
wordnet=WordNetLemmatizer()
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#ps=PorterStemmer()
corpus=[]
for i in range(0,len(message)):
     review=re.sub('[^a-zA-Z]',' ',message['message'][i])
     review=review.lower()
     review=review.split()
     review=[wordnet.lemmatize(word)for word in review if not word in stopwords.words('english')]
     review=' '.join(review)
     corpus.append(review)
     
     
#creating bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(message['label'])
y=y.iloc[:,1].values
             

#train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=0)    
                     


from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer=TfidfTransformer().fit(X_train)
X_train_tf=tf_transformer.transform(X_train)
X_train_tf.shape
tfidf_transformer=TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train)
X_train_tfidf.shape


#training the model using naive)bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train_tfidf,y_train)



y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

