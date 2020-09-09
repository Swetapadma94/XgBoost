#!/usr/bin/env python
# coding: utf-8

# In[21]:


data=pd.read_csv(r'E:\Krish naik\python code\Hyperparameter optimization for xgboost\churn.csv',encoding='latin1')


# In[22]:


data.head()


# In[23]:


## Correlation
import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[24]:


X=data.iloc[:,3:13]
Y=data.iloc[:,13]


# In[25]:


data.columns


# In[26]:


Geography=pd.get_dummies(X['Geography'],drop_first=True)


# In[27]:


Geography.head()


# In[28]:


gender=pd.get_dummies(X['Gender'],drop_first=True)


# In[29]:


gender.head()


# In[30]:


X=X.drop(['Gender','Geography'],axis=1)


# In[31]:


X


# In[32]:


X=pd.concat([X,Geography,gender],axis=1)


# In[33]:


X


# In[34]:


## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[35]:


## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost


# In[36]:


# it is used to record how much time it is taking to excute
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[37]:


classifier=xgboost.XGBClassifier()


# In[38]:


###Verbose- To type the message like how it is working how much time it will take and all.
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[39]:


from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,Y)
timer(start_time) # timing ends here for "start_time" variable


# In[40]:


X.head()


# In[41]:


random_search.best_estimator_


# In[42]:


random_search.best_params_


# In[47]:



classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)


# In[48]:


from sklearn.model_selection import cross_val_score


# In[49]:


score=cross_val_score(classifier,X,Y,cv=10)


# In[50]:


score.mean()


# In[ ]:




