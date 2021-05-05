#!/usr/bin/env python
# coding: utf-8

# ###  Importing Libraries and Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("D:\Ishty Folder\Projects Python\Chronic Kidney Disease\kidney_disease.csv")
data.head()


# ### Changing the columns names using the text description file

# In[3]:


columns = pd.read_csv("D:\Ishty Folder\Projects Python\Chronic Kidney Disease\data_description.txt",sep="-")
columns=columns.reset_index()
columns.columns=["cols","Full_names"]
columns.head()


# In[4]:


data.columns = columns["Full_names"].values
data.head()


# ### Changing the Data Types of Required Columns 

# In[5]:


data.dtypes


# In[6]:


from Function_File import *
features =['packed cell volume','white blood cell count','red blood cell count']
for feature in features:
    convert_dtypes(data,feature)


# In[7]:


data.dtypes


# ### Data Cleaning 

# In[8]:


data.drop('id',axis=1,inplace=True)
data.head()


# In[9]:


from Function_File import *
cat_col,num_col=extract_cat_num(data)


# In[10]:


for col in cat_col:
    print(' {} has {} numbers of values i.e. {}'.format(col, data[col].nunique(),data[col].unique()))


# In[11]:


data["diabetes mellitus"]= data['diabetes mellitus'].replace(to_replace ={'\tno':'no','\tyes':'yes'})
data["coronary artery disease"]= data['coronary artery disease'].replace(to_replace ={'\tno':'no','\tyes':'yes'})
data['class'] = data['class'].replace(to_replace={'ckd\t':'ckd'})
for col in cat_col:
    print(' {} has {} numbers of values i.e. {}'.format(col, data[col].nunique(),data[col].unique()))


# In[12]:


data.isna().sum().sort_values(ascending =False)


# In[13]:


from Function_File import *
data[num_col].isnull().sum()
for f in num_col:
    random_value_imputation(data,f)


# In[14]:


from Function_File import *
data[cat_col].isnull().sum()
for f in cat_col:
    random_value_imputation(data,f)


# ### Data Visualization

# In[15]:


plt.figure(figsize=(30,20))

for i,feature in enumerate(num_col):
    plt.subplot(5,3,i+1)
    data[feature].hist()
    plt.title(feature)


# Conclusion 1: The age, sodium, haemoglobin and packed cell volume are little bit left skewed. Means there are some negative / low value outliers
# Whereas, white blood cell count, senum creatinine, BP, blood urea and blood glucose are right skewed means it is right means high positive outliers.

# In[16]:


plt.figure(figsize=(30,20))
for i,feature in enumerate(cat_col):
    plt.subplot(4,3,i+1)
    sns.countplot(data[feature])


# Since, the features are having imbalance situations hence, cor-relations are needed to be find. 

# In[17]:


plt.figure(figsize=(18,8)) 
sns.heatmap(data.corr(),annot=True)


# In[18]:


get_ipython().system('pip install plotly')


# In[19]:


from Function_File import *
num_col
for i in num_col:
    kdeplot(data,i) 


# Conclusion 2:
# 1. Age group of 50-70 is more likely to be suffering from ckd whereas age group between 20-80 are non ckd. The variance in non ckd data is more than the ckd. 
# 2. The BP for ckd is more variant and lie between 60-100 compared to non ckd 
# 3. The albumin content of non ckd lies less than 1 whereas the albumin content of ckd is more dispersed upto 5. 
# 4. The blood glucose level for non ckd lies 50 to 150 whereas less ckd suffers are having level in this normal range. Ckd sefferers are having more dispersed data. 
# 5. Similarly, for blood urea, only few ckd sufferers are having blood urea in normal range as non ckd sufferers.
# 6. Maximum ckd sufferers are having serum creatinine in range of non ckd sufferers. 
# 7. The sodium and potassium levels of ckd and non ckd are in approximate same range 
# 8. The haemoglobin levels for non ckd is more in range of 12.5-17.5 than ckd in range of 5-17.5 having dispersed data
# 9. The packed cell volume of ckd lies low in normal range i.e. 40-60.
# 10. The wwhite blood count of non ckd and ckd are in approximately same range with approximately same dispersed data. 
# 11. The red blood count of non ckd is more in range of 4-7 whereas ckd sufferers are having less red blood count in range of 3-6

# In[20]:


#To study the RBC Count with respect to different features grouped with class
Feature_Study = input(" Enter the Feature (Having object dtype preferably) to study with red blood cell count:  ")
for i in data.columns:
    if i==Feature_Study:
        print(data.groupby([i,'class'])['red blood cell count'].agg(['count','mean','median','min','max']), end='\n \n \n')


# In[21]:


#To study the WBC Count with respect to different features grouped with class
Feature_Study = input(" Enter the Feature (Having object dtype preferably) to study with white blood cell count:  ")
for i in data.columns:
    if i==Feature_Study:
        print(data.groupby([i,'class'])['white blood cell count'].agg(['count','mean','median','min','max']), end='\n \n \n')


# In[22]:


get_ipython().system('pip install tabulate')
from tabulate import tabulate


# In[23]:


g=[['Name','count','mean','median','min','max']]
for i in data.columns:
    if data[i].dtypes=='float64':
        d=[i, data[i].count(), data[i].mean(), data[i].median(), data[i].min(), data[i].max()]
        g.append(d)      
print(tabulate(g, headers='firstrow'))


# ### Selecting Best Feature From the List

# In[24]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in cat_col:
    data[col]=le.fit_transform(data[col])
data.head()
#0 is ckd and 1 is non ckd


# In[25]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[26]:


ind_col=[col for col in data.columns if col!='class']
dep_col='class'


# In[27]:


x=data[ind_col]
y=data[dep_col]


# In[28]:


ordered_rank_feature=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_feature.fit(x,y)
ordered_feature.scores_


# In[29]:


data_score=pd.DataFrame(ordered_feature.scores_,columns=['score'])


# In[30]:


data_x=pd.DataFrame(x.columns,columns=['feature'])
features_rank=pd.concat([data_x,data_score],axis=1)
features_rank.head()


# In[31]:


features_rank['score'].max()


# In[32]:


features_rank.nlargest(10,'score')


# In[33]:


selected_columns=features_rank.nlargest(10,'score')['feature'].values
selected_columns


# In[34]:


x_new=data[selected_columns]
x_new.head()


# In[35]:


len(x_new)


# In[36]:


x_new.shape


# ### Prediction Model

# In[37]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x_new,y,random_state=0,test_size=0.25)


# In[38]:


ytrain.value_counts() #Balanced Data


# In[39]:


get_ipython().system('pip install xgboost')

from xgboost import XGBClassifier


# In[40]:


classifier=XGBClassifier()


# In[41]:


param={'learning_rate':[0.05,0.20,0.25],
       'max_depth':[5,8,10],
        'min_child_weight':[1,3,5,7],
         'gamma':[0.0,0.1,0.2,0.4],
         'colsample_bytree':[0.3,0.4,0.7]}


# In[42]:


from sklearn.model_selection import RandomizedSearchCV
random_search=RandomizedSearchCV(classifier,param_distributions=param, n_iter=5, scoring='roc_auc', n_jobs=-1,cv=5,verbose=3)


# In[43]:


random_search.fit(xtrain,ytrain)


# In[44]:


random_search.best_estimator_


# In[45]:


random_search.best_params_


# ### Model Initialization

# In[46]:


classifier=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.4, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.05, max_delta_step=0, max_depth=10,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)


# In[47]:


classifier.fit(xtrain,ytrain)


# In[48]:


ypred=classifier.predict(xtest)
ypred


# In[49]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(ytest,ypred)


# In[50]:


accuracy_score(ytest,ypred)


# In[51]:


from sklearn import metrics

print("Accuracy score:",metrics.accuracy_score(ytest, ypred))
print("Precision score:",metrics.precision_score(ytest, ypred))
print("Recall score:",metrics.recall_score(ytest, ypred))
print("F1 Score :",metrics.f1_score(ytest, ypred))


# In[ ]:




