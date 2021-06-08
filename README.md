# Kidney-Disease-Analysis
Chronic kidney disease (CKD) is the 16th leading cause of years of life lost worldwide. Appropriate screening, diagnosis, and management by primary care clinicians are necessary to prevent adverse CKD-associated outcomes, including cardiovascular disease, end-stage kidney disease, and death. Chronic kidney disease (CKD) affects between 8% and 16% of the population worldwide and is often underrecognized by patients and clinicians.

Here, machine learning algorithm is used to detect whether the person will suffer from chronic kidney disease or not. CSV file is attached which contains all the training and test datasets. 
Chronic kidney disease depends upon various factors like, specific gravity, albumin, sugar, red blood cells, pus cell, bacteria and many more.  
Initially, data pre-processing has been done which involved the changing the names the columns, data cleaning like removing the nan values, data visualisation to study the skewness, outliers, balance or imbalance data, corelation and distribution of data. 
After this, best feature selection is done using using SelectKBest and chi2 after applying the label encoding for categorical data. This lead to selection of only top 10 columns based on their feature scores. Function file is made sperataely which is called later in main file. 
Using the XGBoost algorithm, prediction of model is done. And randomised search CV is used to hypertune the parameters. Thus, final model initialisation leads following score, 

Accuracy score: 0.96 , Precision score: 0.925, Recall score: 0.9736842105263158 and F1 Score : 0.9487179487179489

#### Libraries Used:

Pandas, Numpy, Matplotlib, Seaborn, Plotly

#### Programing Language 
Python 

#### IDE Used
Jupyter Notebook

#### Algorthms Used
Label Encoder, SelectKBest, Chi2, XGB Classifier, Randomised Search CV
