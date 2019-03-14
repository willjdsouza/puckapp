#!/usr/bin/env python
# coding: utf-8

# # NHL - Predicting Wins

# ### Overview
# 
# One of my previous portfolio pieces include an analysis of the NHL since 1964 to present. When I first undertook the piece I conducted the analysis using R, and it was done in my early days in data science. It has used both supervised and unsupervised machine learning methods for the predictive analytic portion.
# 
# I am now redoing this piece, and this is just the first part of a multi faceted project that I am working on. It is my most favorite data to work with because of my love for hockey. Analytics have proven successful in other sports like baseball, however I have always believed that due to the "bounce of the puck on a slippery surface" it is VERY difficult to predict an outcome. That is my personal take on it, however, I want to prove this wrong and show that we are able to provide directionality thoroughout each part of this project. 
# 
# For this project I zeroed in from 1997 to present. This portion looks at predicting wins, so that I may be able to  help the number to give insight on the number of wins a team may finish with in a season. 

# #### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor


# #### Uploading data and visualizing it

# In[2]:


np.random.RandomState(40)
teams = pd.read_csv("C:/Users/willjdsouza/windows_stuff/Desktop/hockey_final.csv", encoding='latin-1')


# In[3]:


teams.head()


# In[4]:


teams.shape


# In[5]:


teams.describe()


# They are are a number of ways I looked at the data, however I wanted to share a few visualizations I created with Kibana. I am becoming a huge fan of the elasticstack and am looking to work with it more. The visualizations are meant to be interactive, but I had to share static screenshots because it is quite difficult to share the interactive visualizations
# 
# The first 4 charts I looked at the divisions as a whole and measured different metrics. The next 4 I looked at certain metrics given the teams in a division

# <img src="Division.W_GP.PNG">
# <img src="Division.L_GP.PNG">
# <img src="Division.OTW.PNG">
# <img src="Division.P_GP.PNG">

# <img src="Team.MET.PNG">
# <img src="Team.CEN.PNG">
# <img src="Team.PAC.PNG">
# <img src="Team.ATL.PNG">

# The major point to share after looking at these visualizations can be summed up in one major point. The data is not only well distributed, it is also almost identical when comparing different groups. We dont see many spikes or outliers in the data which means that our predictions may come out quite well. 
# 
# When we segment the data, the magnitude of variances amongst the groups do not deviate from eachother greatly, so we can also that the leauge is highly competitive. Teams average out to having similar statistics, which points out how effective the salary cap system and talent is distributed amongst the teams in the leauge.

# #### Correlations & training data

# In[6]:


teams.corr(method='pearson')


# In[15]:


X = teams[['GF/GP', 'GA/GP', 'PP%', 'PK%', 'S/GP']] #features for training
y = teams.loc[:, ['W']] #target variable

X_train_prepared, X_test_prepared, y_train, y_test,  = train_test_split(X, y, test_size=0.15, random_state=42) #random split for train and test sets

print("X_train length:", len(X_train_prepared))
print("X_test length:", len(X_test_prepared))
print("y_train length:", len(y_train))
print("y_test length:", len(y_test))


# In[13]:


X_train_prepared.dtypes


# In[8]:


from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
    


# In[9]:


#Building a full pipeline for data transformation

#numerical_values = list(teams[['GF', 'GF/GP', 'GA/GP', 'PP%', 'PK%', 'S/GP']])

#catergorical_values = teams[['Team', 'Divison']]

#num_pipeline = Pipeline([
  #      ('selector', DataFrameSelector(numerical_values)),
 #       ('std_scaler', StandardScaler()), #scaling data using a standard scaler
#    ])

#cat_pipeline = Pipeline([
#        ('selector', DataFrameSelector(catergorical_values)),
#        ('cat_encoder', OneHotEncoder(sparse=)),
#    ]) 


#full_pipeline = FeatureUnion(transformer_list=[
#        ("num_pipeline", num_pipeline),
#        ("cat_pipeline", cat_pipeline),
#])


# In[10]:


#from sklearn.pipeline import FeatureUnion

#X_train_prepared = full_pipeline.fit_transform(X_train) 
#X_test_prepared = full_pipeline.transform(X_test)


# In[11]:


#X_train_prepared


# ##### Shortlisting models 

# In[12]:


lin_r = LinearRegression()
lin_r.fit(X_train_prepared, y_train)
lin_r_predict = lin_r.predict(X_train_prepared)
lin_r_scores = mean_squared_error(lin_r_predict, np.ravel(y_train))



las = Lasso()
las.fit(X_train_prepared, y_train)
las_predict = las.predict(X_train_prepared)
las_scores = mean_squared_error(las_predict, np.ravel(y_train))


sgd = SGDRegressor()
sgd.fit(X_train_prepared, y_train)
sgd_predict = sgd.predict(X_train_prepared)
sgd_scores = mean_squared_error(sgd_predict, np.ravel(y_train))


dtr = DecisionTreeRegressor()
dtr.fit(X_train_prepared, y_train)
dtr_predict = dtr.predict(X_train_prepared)
dtr_scores = mean_squared_error(dtr_predict, np.ravel(y_train))


rfr = RandomForestRegressor()
rfr.fit(X_train_prepared, y_train)
rfr_predict = rfr.predict(X_train_prepared)
rfr_scores = mean_squared_error(rfr_predict, np.ravel(y_train))


ada_rfr = AdaBoostRegressor(rfr)
ada_rfr.fit(X_train_prepared, y_train)
ada_rfr_predict = ada_rfr.predict(X_train_prepared)
ada_rfr_scores = mean_squared_error(ada_rfr_predict, np.ravel(y_train))


ada_dtr = AdaBoostRegressor(dtr)
ada_dtr.fit(X_train_prepared, y_train)
ada_dtr_predict = ada_dtr.predict(X_train_prepared)
ada_dtr_scores = mean_squared_error(ada_dtr_predict, np.ravel(y_train))

print("Linear Regression Scores - R2 Score:", r2_score(lin_r_predict, np.ravel(y_train)), "RMSE:", 
      np.sqrt(lin_r_scores))

print("Lasso Scores - RMSE:", np.sqrt(las_scores))
      
print("SGD Scores - RMSE:", np.sqrt(lin_r_scores))

print("Decision Tree Scores - RMSE:", np.sqrt(dtr_scores))

print("Random Forest Scores - RMSE:", np.sqrt(rfr_scores))

print("Adaboost (Random Forest) Scores - RMSE:", np.sqrt(ada_rfr_scores))

print("Adaboost (Decision Trees) Scores - RMSE:", np.sqrt(ada_dtr_scores))


# During the shortlisting phase, it looks like all models do fairly well. The decision trees fit the data tremendously well, however, I can easily tell that this will cause overfitting. 

# #### Hypertuning random forest

# In[13]:



from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [8, 10, 12,],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf':[1, 3, 5]}
  ]

rfrc = RandomForestRegressor()
grid_search = GridSearchCV(rfr, param_grid,
scoring='neg_mean_squared_error')
grid_search.fit(X_train_prepared, np.ravel(y_train))
print(grid_search.best_params_)
print(np.sqrt(-grid_search.best_score_))


# After many attempts at hypertuning, it was difficult to find any sort of gain with it. Turns out the default paramaeters itself work out nicely.

# #### Test set and comaprisons

# Next, I compared the variancse between 4 regressors (random forest, decision tree, linear regression and adaboost). I wanted to see which regressor generalized well so I could choose the best one

# In[14]:


rfr_predict = rfr.predict(X_test_prepared)
rfr_test_scores = mean_squared_error(rfr_predict, np.ravel(y_test))

dtr_predict= dtr.predict(X_test_prepared)
dtr_test_scores = mean_squared_error(dtr_predict, np.ravel(y_test))
    
lin_r_predict = lin_r.predict(X_test_prepared)
lin_r_test_scores = mean_squared_error(lin_r_predict, y_test)
    
ada_dtr_predict = ada_dtr.predict(X_test_prepared)
ada_dtr_test_scores = mean_squared_error(ada_dtr_predict, np.ravel(y_test))
    
ada_rfr_predict = ada_rfr.predict(X_test_prepared)
ada_rfr_test_scores = mean_squared_error(ada_rfr_predict, np.ravel(y_test))
    
d = pd.DataFrame({'Method': ['Linear Regression', 'Decision Trees', 'Adaboost (Decision Trees)',
                             'Random Forest', 'Adaboost (Random Forest)'], 
                  'Training Scores': [np.sqrt(lin_r_scores), np.sqrt(dtr_scores), np.sqrt(ada_dtr_scores), 
                                      np.sqrt(rfr_scores), np.sqrt(ada_rfr_scores)], 
                  'Testing Scores' : [np.sqrt(lin_r_test_scores), np.sqrt(dtr_test_scores), 
                                      np.sqrt(ada_dtr_test_scores), np.sqrt(rfr_test_scores), 
                                      np.sqrt(ada_rfr_test_scores)]
                 })
    
d['Variance'] = d['Testing Scores'] - d['Training Scores']

d


# In my opinion, it seems like Linear Regression or Random Forest is the way to go. Not only did they score the highest in the test set but the variance between the models training and testing set was minimal. If using a randomized search, it may be possible to find parameters that best fit the random forest model and be able to better score than the linear regression.
# 
# 
# For my next part of this project I will be looking more closely on the player level, and will either use TensorFlow or Keras for the machine learning portion.
# 

# In[ ]:


import pickle
with open('hockey_model.pkl', 'wb') as file:
    pickle.dump(lin_r, file)


# In[ ]:




