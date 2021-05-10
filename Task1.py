#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation #GRIPMAY2021
# <!-- Task 1 - Prection using supervised ML
# By Siji Wilson -->

# In[3]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  


# In[4]:


# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")
s_data.head(26)


# In[5]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # Preparing the data

# In[6]:


X = s_data.iloc[:, :-1].values  
Y = s_data.iloc[:, 1].values


# In[7]:


from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                            test_size=0.2, random_state=0)


# # Training the Algorithm

# In[9]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, Y_train) 
print("Training complete.")


# In[11]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# # Making Predictions

# In[15]:


print(X_test) # Testing data - In Hours
Y_pred = regressor.predict(X_test) # Predicting the scores


# In[16]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
print(df)


# In[18]:


# Testing
hours = [[9]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # no. of Hours is 9 and predicted score is 91.21406836721482
