#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df = pd.read_csv('car data.csv')


# In[5]:


df.head()


# In[6]:


df.shape


# In[8]:


print(df['Seller_Type'].unique())


# In[9]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[11]:


## Check missing or null values
df.isnull().sum()


# In[12]:


df.describe()


# In[13]:


df.columns


# In[14]:


final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[15]:


final_dataset.head()


# In[16]:


final_dataset['Current_Year'] = 2020


# In[17]:


final_dataset.head()


# In[19]:


final_dataset['no_year'] = final_dataset['Current_Year']-final_dataset['Year']


# In[20]:


final_dataset.head()


# In[22]:


final_dataset.drop(['Year'],axis = 1,inplace=True)


# In[23]:


final_dataset.drop(['Current_Year'],axis = 1,inplace=True)


# In[24]:


final_dataset.head()


# In[25]:


final_dataset = pd.get_dummies(final_dataset,drop_first = True)


# In[26]:


final_dataset.head()


# In[27]:


final_dataset.corr()


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


sns.pairplot(final_dataset)


# In[34]:


corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
# Plot heat map
g = sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[35]:


final_dataset.head()


# In[36]:


# Independent and dependent features
X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]


# In[37]:


X.head()


# In[38]:


y.head()


# In[43]:


### Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)


# In[44]:


print(model.feature_importances_)


# In[46]:


# Plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_,index = X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[47]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)   ## 20% id test data


# In[48]:


X_train.shape


# In[ ]:


# from sklearn.ensemble import RandomForestRegressor
# rf_random = RandomForestRegressor()


# In[50]:


# ### Hyperparameters
# import numpy as np
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# print(n_estimators)


# In[51]:


#from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


# #Randomized Search CV

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# # max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10, 15, 100]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 5, 10]


# In[52]:


from sklearn.linear_model import LinearRegression


# In[53]:


model = LinearRegression()
model.fit(X_train,y_train)


# In[55]:


pred_y = model.predict(X_test)
print(pred_y)


# In[56]:


plt.plot(X_test,pred_y)


# In[58]:


print(pd.DataFrame(pred_y).head())


# In[59]:


print(y_test.head())


# In[60]:


plt.plot(y_test,pred_y)


# In[67]:


plt.plot(X_test,pred_y,'o')
plt.plot(X_train,y_train,'o')


# In[69]:


print(y_test.shape)
print(pred_y.shape)


# In[65]:


# print("Accuracy Score :")
# print(accuracy_score(y_test,pred_y))
# print("Classification Report :")
# print(classification_report(y_test,pred_y))


# In[70]:


sns.distplot(y_test-pred_y)


# In[72]:


plt.scatter(y_test,pred_y)


# In[64]:


# calculate MAE, MSE, RMSE
from sklearn import metrics
print(metrics.mean_absolute_error(y_test,pred_y))
print(metrics.mean_squared_error(y_test,pred_y))
print(np.sqrt(metrics.mean_squared_error(y_test,pred_y)))


# In[ ]:




