##############################################

data={"name":["a","b","c","d","e","f"],
      "age":[1,2,3,4,5,6],
      "edu":["pg","ug","phd","phd","pg","ug"]}
#convert inot dataframe
import pandas as pd
df=pd.DataFrame(data)
print(df)
#####################################
name=["a","b","c","d","e","f"]
age=[1,2,3,4,5,6]
e=list(zip(name,age))
df=pd.DataFrame(e)
print(df)
#####################################
data_sal=pd.read_csv(r"C:\Users\Saurabh\Desktop\Salaries.csv")

data_sal.describe()
data_sal.columns
data_sal.get_dtype_counts()
data_sal.get_value
### hw#######
data_sal.merge()
####hw######
data_sal.nunique()
######################################
#fill missing value numeric
data_sal.isnull().sum()# check missing value
###basepay analyis
data_sal["BasePay"].plot.hist()

#we fill value according to data distribution
data_sal["BasePay"].fillna(data_sal["BasePay"].median(),inplace=True)

data_sal["BasePay"].median()
data_sal.isnull().sum()
####drop colms
data_sal.pop("Status")
data_sal.isnull().sum()
######################
data_pre=data_sal.drop("Notes",axis=1)
data_pre.isnull().sum()
######################
data_pre.info()
data_pre.isnull().sum()



#######################################
#dummies########
jt=pd.get_dummies(data_pre["JobTitle"])
jt.head()


data={"name":["a","b","c","d","e","f"],
      "age":[1,2,3,4,5,6],
      "edu":["pg","ug","phd","phd","pg","ug"],
      "gender":["male","female","female","person","female","female"]}
#convert inot dataframe
import pandas as pd
df=pd.DataFrame(data)
print(df)
##################################

g=pd.get_dummies(df["gender"],drop_first=True)
g.head()

######user choice column

g=pd.get_dummies(df["gender"])
g.pop("person")
g.head()
##concat two data frame
data=pd.concat([df,g],axis=1)

#####################
data1=pd.concat([df,g],axis=0)

################################
edu=pd.get_dummies(df["edu"],drop_first=True)
data_pre=pd.concat([data,edu],axis=1)

######drop multiple cols
data_final=data_pre.drop(["edu","gender"],axis=1)
data_final