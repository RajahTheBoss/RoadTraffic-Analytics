#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pprint import pprint
import glob
import codecs
import json
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)


# ## 1.1 Data parsing

# In[2]:


read_files = glob.glob("data/*.geojson")
output_list = []

for f in read_files:
    with open(f, "rb") as infile:
        output_list.append(json.load(infile))

with open("merged_file.json", "w") as outfile:
    json.dump(output_list, outfile)


# In[3]:


with codecs.open('merged_file.json', 'r', 'utf-8') as json_file:  
    data = json.load(json_file)
    
df = pd.json_normalize(data, errors='ignore')
df


# In[4]:


df = pd.json_normalize(data, record_path=['features', 'properties', 'vehicles', 'participants'], meta = [
    ['features', 'properties','id'],
    ['features', 'properties', 'tags'],
    ['features', 'properties', 'light'],
    ['features', 'properties', 'point'], #2 cols
    ['features', 'properties', 'nearby'],
    ['features', 'properties', 'region'],
    ['features', 'properties', 'address'],
    ['features', 'properties', 'weather'],
    ['features', 'properties', 'category'],
    ['features', 'properties', 'datetime'],
    ['features', 'properties', 'severity'],
    ['features', 'properties', 'vehicles', 'year'],
    ['features', 'properties', 'vehicles', 'brand'],
    ['features', 'properties', 'vehicles', 'color'],
    ['features', 'properties', 'vehicles', 'model'],
    ['features', 'properties', 'vehicles', 'category'],
    ['features', 'properties','dead_count'],
    ['features', 'properties','participants'],
    ['features', 'properties','injured_count'],
    ['features', 'properties','parent_region'],
    ['features', 'properties','road_conditions'],
    ['features', 'properties','participants_count'],
    ['features', 'properties','participant_categories'],
], errors='ignore')

df = pd.concat([df.drop('features.properties.point', axis=1), pd.DataFrame(df['features.properties.point'].tolist())], axis=1)
df


# In[5]:


df1 =  (df.set_index('features.properties.id')['features.properties.participants']
       .apply(pd.Series).stack()
         .apply(pd.Series).reset_index().drop('level_1',1))

df = df.merge(df1, how='left', on='features.properties.id')


# In[6]:


df=df.drop('features.properties.participants', axis=1)


# Let's check empty values

# In[7]:


df.isna().sum()


# ## 1.2 Data preprocessing and highlighting of significant attributes

# All data is uploaded and json is presented. Consider some statistics and dimension

# In[8]:


df=df.fillna(0)


# In[9]:


df.shape


# In[10]:


df.info()


# Let's check the number of empty values after preprocessing

# In[11]:


df.isna().sum()


# ### Preprocessing enumerations in a dataset

# In[12]:


df = df.explode('violations_x')
df = df.explode('features.properties.tags')
df = df.explode('features.properties.nearby')
df = df.explode('features.properties.weather')
df = df.explode('features.properties.road_conditions')
df = df.explode('features.properties.participant_categories')


# In[13]:


df


# In[14]:


df=df.drop_duplicates(subset=['features.properties.id'])
df=df.fillna(0)


# In[15]:


df=df[df['features.properties.address']!=0]
df.reset_index(drop=True, inplace=True)


# In[16]:


result = pd.merge(df, df.groupby(['features.properties.address']).size().sort_values(ascending=False).to_frame(), on="features.properties.address")
result.rename(columns={0: 'count'},inplace=True)
#result.groupby(['properties.address']).size().sort_values(ascending=False).to_frame()
df = result


# ### Determining the most important attributes
# To find the most significant attributes, let's build the Pearson correlation on the heat map

# In[17]:


corr=df.drop(['features.properties.id'], axis=1).corr()
plt.figure(figsize=(16, 16))

heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=20)


# As we can see above, the most significant attributes are: `features.properties.dead_count`, `features.properties.injured_count`, `features.properties.participants_count`

# ## 1.3 Description of the data set structure
# 
# * "id": 384094, # id
# * "tags": ["Traffic accidents"], # indicators from the official website of the traffic police
# * "light": "Daylight time", # time of day
# * "point": {"lat": 50.6039, "long": 36.5578}, # coordinates
# * "nearby": [ "Unregulated intersection of unequal streets (roads)", "Individual residential buildings"], # coordinates
# * "region": "Belgorod", # city/district
# * "address": "Belgorod, Sumskaya str., 30", # address
# * "weather": ["Clear"], # weather
# * "category": "Collision", # type of accident
# * "datetime": "2017-08-05 13:06:00", # date and time
# * "severity": "Light", # severity of an accident/harm to health
# * "vehicles": [ # participants – vehicles
# *
# * "year": 2010, # year of vehicle production
# * "brand": "VAZ", # vehicle brand
# * "color": "Other colors", # vehicle color
# * "model": "Priora", # vehicle model
# * "category": "C-class (small medium, compact) up to 4.3 m", # vehicle category
# * "participants": [ # participants inside vehicles
# *
# * "role": "Driver", # participant role
# * "gender": "Female", # gender of the participant
# * "violations": [], # violations of the rules by the participant
# * "health_status": "Injured, being...", # health status of the participant
# * "years_of_driving_experience": 11 # participant's driving experience (drivers only)
# 
# * "dead_count": 0, # number of people killed in an accident
# * "participants": [], # participants without vehicles (description as participants inside vehicles)
# * "injured_count": 2, # number of injured in an accident
# * "parent_region": "Belgorod region", # region
# * "road_conditions": ["Dry"], # road surface condition
# * "participants_count": 3, # number of road accident participants
# * "participant_categories": ["All participants", "Children"] # categories of participants

# In[18]:


df.isna().sum()


# ## 1.4 Formation of additional attributes
# The index will be formed based on the number of accidents, frequency and severity.

# In[19]:


df['Hazard_level'] = None
count_places_max = df['count'].max()
injured_max = df['features.properties.injured_count'].max()
dead_max = df['features.properties.dead_count'].max()


# In[20]:


for i in range(len(df)):
    if df['features.properties.dead_count'][i] > 0:
        df['Hazard_level'][i] = (df['features.properties.injured_count'][i]+df['count'][i])/((injured_max+count_places_max)/2)/4
    else:
        df['Hazard_level'][i] = (df['features.properties.dead_count'][i]*100/dead_max)/100/2+0.5


# In[21]:


df.head()


# In[22]:


df.to_csv('result_data.csv', encoding='utf-8-sig', index=False)


# ## Report
# 
# * 1.1 Data parsing - Data downloaded from the data folder
# * 1.2 Data Preprocessing and highlighting of significant attributes - Data is preprocessed and the most significant attributes are highlighted
# * 1.3 Description of the data set structure - a description is provided for each attribute
# * 1.4 Formation of additional attributes - an additional index based on the data generated
