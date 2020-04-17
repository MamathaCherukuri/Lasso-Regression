#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[ ]:


# difference of lasso and ridge regression is that some of the coefficients can be zero i.e. some of the features are 
# completely neglected


# In[2]:


from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# In[4]:


cancer = load_breast_cancer()
print(cancer.keys())


# In[5]:


cancer_df=pd.DataFrame(cancer.data,columns=cancer.feature_names)


# In[6]:


cancer_df.head()


# In[7]:


cancer_df.info()


# In[8]:


X=cancer_df
Y=cancer.target


# In[9]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=31)


# In[13]:


print("length of train:",len(X_train))
print("length of target train:",len(Y_train))
print("length of test:",len(X_test))
print("length of target test:",len(Y_test))


# In[14]:


lasso=Lasso()
lasso.fit(X_train,Y_train)
train_score=lasso.score(X_train,Y_train)
test_score=lasso.score(X_test,Y_test)


# In[15]:


coeff_used = np.sum(lasso.coef_!=0)


# In[17]:


print("training score:", train_score)
print("test score: ", test_score)
print("number of features used: ", coeff_used)


# In[19]:


lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train,Y_train)


# In[20]:


train_score001=lasso001.score(X_train,Y_train)
test_score001=lasso001.score(X_test,Y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)


# In[21]:


print("training score for alpha=0.01:", train_score001) 
print("test score for alpha =0.01: ", test_score001)
print("number of features used: for alpha =0.01:", coeff_used001)


# In[22]:


lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(X_train,Y_train)
train_score00001=lasso00001.score(X_train,Y_train)
test_score00001=lasso00001.score(X_test,Y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)
print("training score for alpha=0.0001:", train_score00001) 
print("test score for alpha =0.0001: ", test_score00001)
print("number of features used: for alpha =0.0001:", coeff_used00001)


# In[23]:


lr = LinearRegression()
lr.fit(X_train,Y_train)
lr_train_score=lr.score(X_train,Y_train)
lr_test_score=lr.score(X_test,Y_test)
print("LR training score:", lr_train_score)
print("LR test score: ", lr_test_score)


# In[24]:


plt.subplot(1,2,1)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.subplot(1,2,2)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




