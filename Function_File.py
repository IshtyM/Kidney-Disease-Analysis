#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def convert_dtypes(data,feature):
    data[feature]=pd.to_numeric(data[feature],errors='coerce')


# In[3]:


def extract_cat_num(data):
    cat_col=[]
    num_col=[]
    for col in data.columns:
        if data[col].dtype=='object':
            cat_col.append(col)
        elif data[col].dtype!='object':
            num_col.append(col)
    return cat_col, num_col


# In[4]:


get_ipython().system('pip install plotly')
import plotly.express as px


# In[5]:


def violin(Data,col):
    fig=px.violin(Data,y=col,x='class',color='class',box=True)
    return fig.show()


# In[6]:


def scatter(Data,col1, col2): 
    fig=px.scatter(Data,x=col1 ,y=col2, color='class')
    return fig.show()


# In[7]:


def kdeplot(Data,col): 
    grid=sns.FacetGrid(Data, hue='class',aspect=2)
    grid.map(sns.kdeplot,col)
    grid.add_legend()


# In[9]:


def random_value_imputation(data,feature):
    random_sample=data[feature].dropna().sample(data[feature].isnull().sum())
    random_sample.index=data[data[feature].isnull()].index
    data.loc[data[feature].isnull(),feature]=random_sample
    print("The missing values",feature," are ", data[feature].isnull().sum())

