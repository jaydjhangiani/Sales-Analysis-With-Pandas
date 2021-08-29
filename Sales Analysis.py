#!/usr/bin/env python
# coding: utf-8

# ## Sales Analysis

# #### Pandas Case Study

# In[1]:


import pandas as pd
import os
import matplotlib.pyplot as plt


from itertools import combinations
from collections import Counter


# #### Task 1: Merging 12 months of sales data into a single file

# In[2]:


df = pd.read_csv("./Sales_Data/Sales_April_2019.csv")

files = [file for file in os.listdir("./Sales_data")]

all_months_data = pd.DataFrame()

for file in files:
    df = pd.read_csv("./Sales_Data/"+file)
    all_months_data = pd.concat([all_months_data, df])
    
all_months_data.to_csv("all_data.csv", index=False)    


# #### Read updated dataframe

# In[3]:


all_data = pd.read_csv("all_data.csv")
all_data.head()


# ### Clean up the data!

# #### Drop NaN rows

# In[4]:


nan_df = all_data[all_data.isna().any(axis=1)]
nan_df.head()

all_data = all_data.dropna(how="all")


# #### Find 'Or' and delete it

# In[5]:


all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']


# #### Convert  columns to correct type

# In[6]:


all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])

all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])


# ### Augment data with additional columns

# #### Task 2: Add Month Column

# In[7]:


all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype('int32')

all_data.head()


# #### Task 3: Add a sales column

# In[8]:


all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']
all_data.head()


# #### Task 4: Add a city column

# In[9]:


# .apply() method

def get_city(address):
    return address.split(',')[1]

def get_state(address):
    return address.split(',')[2].split(' ')[1]

all_data['City'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)})")

all_data.head()


# ##### Question 1: What was the best month for sales? How much was earned that month?

# In[10]:


month_result = all_data.groupby('Month').sum()


# In[11]:


months = range(1,13)
plt.bar(months, month_result['Sales'])
plt.xlabel('Months')
plt.ylabel('Sales in USD ($)')
plt.show()


# ##### Question 2: Which city had the highest number of sales

# In[12]:


city_result = all_data.groupby('City').sum()
city_result


# In[13]:


cities = [city for city ,df in all_data.groupby('City')]

plt.bar(cities, city_result['Sales'])
plt.xticks(cities, rotation="vertical", size=8)
plt.xlabel('City Name')
plt.ylabel('Sales in USD ($)')
plt.show()


# ##### Question 3: What time should we display ads to maximise likelihhod of customer's buying products?

# In[14]:


all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])


# In[15]:


all_data['Hour'] = all_data['Order Date'].dt.hour
all_data['Minute'] = all_data['Order Date'].dt.minute


# In[16]:


hours = [hour for hour, df in all_data.groupby('Hour')]

plt.plot(hours, all_data.groupby('Hour').count() )
plt.xticks(hours)
plt.grid()
plt.show()


# ##### Question 4: What products are most often sold together?

# In[17]:


df = all_data[all_data['Order ID'].duplicated(keep=False)]

df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x:','.join(x) )

df = df[['Order ID', 'Grouped']].drop_duplicates()

df.head(10)


# In[23]:


count = Counter()

for row in df['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list,2)))
    
for key, value in count.most_common(10):
    print(key, value)


# ##### Question 5: Which product sold the most and why?

# In[35]:


product_group = all_data.groupby('Product')

quantity_ordered = product_group.sum()['Quantity Ordered']

products = [product for product , df in product_group]

plt.bar(products, quantity_ordered)
plt.xticks(products, rotation="vertical", size=8)
plt.xlabel('Products')
plt.ylabel('Quantity Ordered ')
plt.show()


# In[50]:


prices = all_data.groupby('Product').mean()['Price Each']

flg, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered, color='g')
ax2.plot(products, prices,'b-')

ax1.set_xlabel('Products')
ax1.set_ylabel('Quantity Ordered', color='g')
ax2.set_ylabel('Price in USD ($)', color='b')
ax1.set_xticklabels(products, rotation='vertical', size=8)

plt.show()


# In[ ]:




