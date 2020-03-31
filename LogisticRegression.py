#!/usr/bin/env python
# coding: utf-8

# # What is Logistic Regression?

# Logistic Regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary).  Like all regression analyses, the logistic regression is a predictive analysis.  Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.

# # Logistic Function

# Logistic regression is named for the function used at the core of the method, the logistic function.
# 
# The logistic function, also called the sigmoid function was developed by statisticians to describe properties of population growth in ecology, rising quickly and maxing out at the carrying capacity of the environment. It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits.
# 
# 1 / (1 + e^-value)
# 
# Where e is the base of the natural logarithms (Euler’s number or the EXP() function in your spreadsheet) and value is the actual numerical value that you want to transform. Below is a plot of the numbers between -5 and 5 transformed into the range 0 and 1 using the logistic function.

# Logistic regression is a linear method, but the predictions are transformed using the logistic function. The impact of this is that we can no longer understand the predictions as a linear combination of the inputs as we can with linear regression, for example, continuing on from above, the model can be stated as:
# 
# p(X) = e^(b0 + b1*X) / (1 + e^(b0 + b1*X))
# 
# I don’t want to dive into the math too much, but we can turn around the above equation as follows (remember we can remove the e from one side by adding a natural logarithm (ln) to the other):
# 
# ln(p(X) / 1 – p(X)) = b0 + b1 * X
# 
# This is useful because we can see that the calculation of the output on the right is linear again (just like linear regression), and the input on the left is a log of the probability of the default class.
# 
# This ratio on the left is called the odds of the default class (it’s historical that we use odds, for example, odds are used in horse racing rather than probabilities). Odds are calculated as a ratio of the probability of the event divided by the probability of not the event, e.g. 0.8/(1-0.8) which has the odds of 4. So we could instead write:
# 
# ln(odds) = b0 + b1 * X
# 
# Because the odds are log transformed, we call this left hand side the log-odds or the probit. It is possible to use other types of functions for the transform (which is out of scope_, but as such it is common to refer to the transform that relates the linear regression equation to the probabilities as the link function, e.g. the probit link function.
# 
# We can move the exponent back to the right and write it as:
# 
# odds = e^(b0 + b1 * X)
# 
# All of this helps us understand that indeed the model is still a linear combination of the inputs, but that this linear combination relates to the log-odds of the default class.

# # Learning the Logistic Regression Model

# Making predictions with a logistic regression model is as simple as plugging in numbers into the logistic regression equation and calculating a result.
# 
# Let’s make this concrete with a specific example.
# 
# Let’s say we have a model that can predict whether a person is male or female based on their height (completely fictitious). Given a height of 150cm is the person male or female.
# 
# We have learned the coefficients of b0 = -100 and b1 = 0.6. Using the equation above we can calculate the probability of male given a height of 150cm or more formally P(male|height=150). We will use EXP() for e, because that is what you can use if you type this example into your spreadsheet:
# 
# y = e^(b0 + b1*X) / (1 + e^(b0 + b1*X))
# 
# y = exp(-100 + 0.6*150) / (1 + EXP(-100 + 0.6*X))
# 
# y = 0.0000453978687
# 
# Or a probability of near zero that the person is a male.
# 
# In practice we can use the probabilities directly. Because this is classification and we want a crisp answer, we can snap the probabilities to a binary class value, for example:
# 
# 0 if p(male) < 0.5
# 
# 1 if p(male) >= 0.5
# 
# Now that we know how to make predictions using logistic regression, let’s look at how we can prepare our data to get the most from the technique.

# # TITANIC DATASET 

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn


from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report


# The first thing we are going to do is to read in the dataset using the Pandas' read_csv() function. We will put this data into a Pandas DataFrame, called "titanic", and name each of the columns.

# In[8]:


data = pd.read_csv("titanic.csv")
#data.columns = ['Survived','Pclass','Name','Sex','Age','Siblings/Spouses Aboard','Parent/child Aboard','Fare']
data.head()


# In[9]:


data.info()


# # VARIABLE DESCRIPTIONS
# Survived - Survival (0 = No; 1 = Yes)
# Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# Name - Name
# Sex - Sex
# Age - Age
# SibSp - Number of Siblings/Spouses Aboard
# Parch - Number of Parents/Children Aboard
# Fare - Passenger Fare (British pound)

# # Checking for missing values
# 
# It's easy to check for missing values by calling the isnull() method, and the sum() method off of that, to return a tally of all the True values that are returned by the isnull() method.

# In[7]:


data.isnull()


# In[8]:


data.isnull().sum()


# In[9]:


data.info()


# So let's just go ahead and drop all the variables that aren't relevant for predicting survival. We should at least keep the following:
# 
#     Survived - This variable is obviously relevant.
#     Pclass - Does a passenger's class on the boat affect their survivability?
#     Sex - Could a passenger's gender impact their survival rate?
#     Age - Does a person's age impact their survival rate?
#     SibSp - Does the number of relatives on the boat (that are siblings or a spouse) affect a person survivability? Probability
#     Parch - Does the number of relatives on the boat (that are children or parents) affect a person survivability? Probability
#     Fare - Does the fare a person paid effect his survivability? Maybe - let's keep it.
#     

# In[10]:


titanic_data = data.drop(['Pclass','Name'], 1)
titanic_data.head()


# # Converting categorical variables to a dummy indicators
# 
# The next thing we need to do is reformat our variables so that they work with the model. Specifically, we need to reformat the Sex and Embarked variables into numeric variables.
# 

# In[11]:


gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)
gender.head()


# In[12]:


titanic_data.head()


# In[13]:


titanic_data.drop(['Sex'],axis=1,inplace=True)
titanic_data.head()


# In[14]:


titanic_dmy = pd.concat([titanic_data,gender],axis=1)
titanic_dmy.head()


# In[15]:


titanic_dmy.drop(['Fare'],axis=1,inplace=True)
titanic_dmy.head()


# In[16]:


X = titanic_dmy.ix[:,(1,2,3,4)].values
y = titanic_dmy.ix[:,0].values


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)


# # Deploying and evaluating the model

# In[18]:


LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)


# In[21]:


y_pred = LogReg.predict(X_test)
y_pred


# In[20]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix


# The results from the confusion matrix are telling us that 137 and 69 are the number of correct predictions. 34 and 27 are the number of incorrect predictions.

# In[31]:


print(classification_report(y_test, y_pred))


# The f1-score gives you the harmonic mean of precision and recall. The scores corresponding to every class will tell you the accuracy of the classifier in classifying the data points in that particular class compared to all other classes. The support is the number of samples of the true response that lie in that class
