#!/usr/bin/env python
# coding: utf-8

# # 1.Preprocessing- <span style="color:red">CATEGORICAL DATA</span>

# ### 1.1 Import libraries

# In[83]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1.2 Import Data

# In[84]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign column names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']


df=pd.read_csv(url,names=names)
df.head(4)


# ### 1.2 To know how many <span style="color:blue">different classes</span> we in columns 'Class' are.

# In[85]:


df['Class'].unique()


# ### 1.3 General Info about DataSet

# In[86]:


df.info()


# ### 1.4 Investigate of NaN Values in columns

# In[87]:


df.isnull().sum()


# ### 1.5 Plotting for general overview of the DataSet

# In[88]:


ax=sns.heatmap(df.corr(),square=True,cmap='YlOrRd')


# In[89]:


g = sns.pairplot(df, plot_kws={'color':'green'})


# # 2. <span style="color:#0E6983">Processing</span>

# ### 2.1 Define of X,y

# In[90]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1:]


# ### 2.2 Define of <span style="color:red">train-test-split</span>

# In[91]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)


# ### 2.3 <span style="color:blue">FeatureScaling</span>

# #### 2.3.1 <span style="color:blue">Call StandrdScaler</span>

# In[92]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# # 2.4 <span style="color:orange">Kneighbors Classifier</span>

# In[96]:


from sklearn.neighbors import KNeighborsClassifier
classifire=KNeighborsClassifier(n_neighbors=5)
classifire.fit(X_train,y_train)


# # 2.5 <span style="color:green">Prediction</span>

# In[98]:


y_pred=classifire.predict(X_test)


# # 2.6 Confusion Matrix

# In[103]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

