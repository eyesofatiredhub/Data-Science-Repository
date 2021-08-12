#!/usr/bin/env python
# coding: utf-8

# # DataCamp Certification Case Study
# 
# ### Project Brief
# 
# You have been hired as a data scientist at a used car dealership in the UK. The sales team have been having problems with pricing used cars that arrive at the dealership and would like your help. Before they take any company wide action they would like you to work with the Toyota specialist to test your idea. They have already collected some data from other retailers on the price that a range of Toyota cars were listed at. It is known that cars that are more than £1500 above the estimated price will not sell. The sales team wants to know whether you can make predictions within this range.
# 
# The presentation of your findings should be targeted at the Head of Sales, who has no technical data science background.
# 
# The data you will use for this analysis can be accessed here: `"data/toyota.csv"`

# In[1]:


# Use this cell to begin, and add as many cells as you need to complete your analysis!
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# The analysis proceeds through the following workflow:
#  Importing Data
#  Exploratory Data Analysis (EDA)
#  Uni-Variate and MutliVariate Analysis
#  Visualization of data, its distribution and central tendencies
#  Exploration of Correlation between variables
#  Preprocessing and Feature Engineering including Variable Transformations
#  Model Fitting
#  Model Evaluation

# In[58]:


import matplotlib
colors = matplotlib.colors.cnames.keys()


# In[3]:


df = pd.read_csv('data/toyota.csv')
df.head()


# In[4]:


df.info()


# Our dataset has no null values.

# In[5]:


df.shape


# In[56]:


df.describe()


# In[38]:


sns.set_style('whitegrid')
sns.distplot(df['price'], color = 'darkblue') #distribution of the target variable


# In[7]:


df['fuelType'].value_counts()


# In[53]:


sns.scatterplot(x = 'mileage', y = 'price', data = df, color = 'limegreen')


# The general trend steeps toward telling us that the increase in mileage decreases the prices. 

# In[9]:


sns.heatmap(df.corr(), annot = True, linewidth = 0.5)


# In[10]:


corr = df.corr()['price'].sort_values(ascending = False)[1:]
sns.barplot(x = corr.index, y = corr, color = 'teal')


# From the above correlation plots, we are able to know the variables that have the most impact on the car prices. So, in order for us to attempt to decrease the sale prices, we have to keep specifically these attributes in mind. The car models are not inlcuded above, so I will now turn to explore the correlation between the models, other variables and car prices. 

# In[11]:


df['model'].value_counts()


# In[12]:


plt.figure(figsize = (20,20))
sns.boxplot(x = 'model', y = 'price', data = df)
plt.xticks(rotation = 90)


# The above visualization shows us the models that tend to be the most expensive. The models 'Land cruiser' and 'Supra' look to be more on the expensive side. And the general central tendencies of their costs.

# In[13]:


sns.scatterplot(x = 'year', y = 'price', data =df, color = 'mediumpurple')


# In[14]:


sns.boxplot(x = 'engineSize', y = 'price', data = df)


# The following lines of code are mostly concerned with preprocessing the data to feed our machine learning model.

# In[15]:


df_dumms = pd.get_dummies(df['model'], drop_first = True)
df_all = df_dumms.join(df)
df_all.drop('model', axis = 1, inplace = True)
df_all


# In[16]:


df_all['transmission'].value_counts()


# In[17]:


df_all['transmissions'] = 0
df_all.loc[df_all['transmission'] == 'Automatic', 'transmissions'] = 0
df_all.loc[df_all['transmission'] == 'Manual', 'transmissions'] = 1
df_all.loc[df_all['transmission'] == 'Semi-Auto', 'transmissions'] = 2
df_all.loc[df_all['transmission'] == 'Other', 'transmissions'] = 3
df_all['transmissions'].value_counts()


# In[18]:


df['fuelType'].value_counts()


# In[19]:


df_dumms2 = pd.get_dummies(df['fuelType'], drop_first = True)
df_all2 = df_dumms2.join(df_all)
df_all2


# In[20]:


df_all2.drop('transmission', axis = 1, inplace = True)


# In[22]:


df_all2.drop('fuelType', axis = 1, inplace = True)


# In[23]:


df_all2


# From here on, we start building and evaluating our machine learning model. I have chosen a regression model here. 

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score


X= df_all2.drop('price', axis = 1)
y = df_all2['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_train.head()


# In[32]:


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** (1/2)
    mae = mean_absolute_error(y_true, y_pred)
    
    print('The mean absolute error is:', mae)
    print('The mean squared error is:', mse)
    print('The root mean squared error is:', rmse)
     


# All of these evaluation metrics are basically different kinds of loss functions. So, we try to build models that minimize them as much as possible.

# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[28]:


y_preds = model.predict(X_test)
y_preds


# In[33]:


evaluate(y_test, y_preds)


# In[35]:


cvs = cross_val_score(model, X, y , cv = 10) 
cvs


# The results of the k-fold cross valuation scores help us to check if we have overfit the data. Since no values are very close to one, we can ce assured that our model has not overfit. We also have relatively high accuracies in each iteration which helps us further to accept our trained model.

# In[36]:


sns.distplot(y_preds)


# In[44]:


plt.scatter(y_test, y_preds, color = 'tomato')
plt.xlabel('True y values')
plt.ylabel('Predicted y values')


# In[45]:


from sklearn.tree import DecisionTreeRegressor

model2 = DecisionTreeRegressor()
model2.fit(X_train, y_train)


# In[46]:


y_preds2 = model2.predict(X_test)


# In[48]:


y_preds2


# In[50]:


evaluate(y_test, y_preds2)


# In[57]:


cvs = cross_val_score(model2, X, y , cv = 10) 
cvs


# In[52]:


sns.distplot(y_preds2)


# In[51]:


plt.scatter(y_test, y_preds2, color = 'tomato')
plt.xlabel('True y values')
plt.ylabel('Predicted y values')


# It is clear through the evaluation metrics that DecisionTreeRegressor is a better suited model. 

# Through the analysis made above, we are able to point out the exact variables that correlate to the high prices of our car sales. We are also able to derive thereafter the best measures to decrease the estimated prices. And, we now also know the better model to use for our predictions. The evaluate function in the notebook can also further be used to evaluate other machine learning models and better our predictions. 
