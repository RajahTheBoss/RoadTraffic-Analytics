#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
pd.set_option('display.max_columns', None)


# In[2]:


df=pd.read_csv('result_data.csv')


# In[5]:


df.head()


# ## 2.1 Data splitting

# Let's split the data set in the way recommended according to the `Sklearn` documentation. Namely, `30 by 70`. As presented in the description, such a sample is optimal, since the absolute majority of data must be found when training the model in order to obtain the most optimized model from the side of its accuracy
# 
# ### Stratification
# When dividing, we stratify the data to get the same percentage of the sample, so that there is no preponderance for any one class and such a situation does not affect incorrect training of the model

# In[6]:


X=df[['features.properties.dead_count', 'features.properties.injured_count', 'features.properties.participants_count']]
y=df['features.properties.severity']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


# ## 2.3 Classification
# 
# 
# ### KNeighborsClassifier
# 
# ### RandomForestClassifier
# 
# ### GaussianNB
# 
# 
# ## Матрикики
# 
# ### accuracy f1-score
# 
# ### macro avg f1-score

# ## 2.4 Training

# In[7]:


from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[8]:


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
preds=neigh.predict(X_test)
print(classification_report(preds, y_test))


# In[9]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_preds=rfc.predict(X_test)
print(classification_report(rfc_preds, y_test))


# In[10]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_preds=gnb.predict(X_test)
print(classification_report(gnb_preds, y_test))


# ### Conclusion
# The most optimal model will be `RandomForestClassifier` with accuracy f1-score = `0.68` and macro avg f1-score = `0.71`, since it showed the best result compared to others.

# ## 3.4 Feature Engineering
# 
# We transform the data set by generating new data in order to increase the accuracy of the classifier and using StandardScaler

# In[13]:


result = pd.merge(df, df.groupby(['features.properties.vehicles.brand']).size().sort_values().to_frame(), on='features.properties.vehicles.brand')
result.rename(columns={0: 'brand_count'}, inplace=True)
df = result


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


scaler = StandardScaler()
X=df[['features.properties.dead_count', 'features.properties.injured_count', 'features.properties.participants_count', 'brand_count']]
y=df['features.properties.severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_preds=rfc.predict(X_test)
print(classification_report(rfc_preds, y_test))


# ## Conclusions on Feature Engineering 
# From the results above, the data conversion for Feature Engineering did not lead to an improvement in the model

# ## Report
# * 2.1 Splitting the data set - the data set is divided into training and test samples
# * 2.3 Classification - 3 classification algorithms and metrics have been selected for their testing
# * 2.4 Training - classification according to the severity of the accident was made
# * 2.5 Feature Engineering - training was performed once again on the transformed data
# 

# In[ ]:


df.to_csv('result_data.csv', encoding='utf-8-sig', index=False)

