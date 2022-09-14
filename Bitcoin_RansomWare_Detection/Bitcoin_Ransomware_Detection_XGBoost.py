# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 00:10:18 2022

@author: Hang
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, auc, accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error


import xgboost as xgb



wd = r'C:\Users\Hahn\Desktop\code\machine learning\bitcoin rs'
os.chdir(wd)



##############################################
# Data Preprocessing
##############################################
## loading data
data_dir = r'data\BitcoinHeistData.csv'
data = pd.read_csv(data_dir)

# summary of descriptive statistics of dataset
data.shape
data.columns
data.describe()
# none nan
data.isnull().sum()


## Feature selection
y = data.label
data_features = ['length', 'weight', 'count', 'looped','neighbors', 'income']
X = data[data_features]
X.describe()
X.head()

## validation
from sklearn.model_selection import train_test_split
#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)




##############################################
## XGBoost Modeling
##############################################
xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
xgb_model.fit(X_train_full, y_train)

# save trained model
xgb_model.save_model("xgb_model.json")
# load trained model
#xgb_model.load_model("xgb_model.json")



##############################################
# Test dataset Prediction
##############################################
y_pred = xgb_model.predict(X_valid)
y_pred = le.inverse_transform(y_pred)

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
# precision, recall, fscore, support
labell = list( np.unique(y_pred) )
prfs = precision_recall_fscore_support(y_valid, y_pred, labels=labell)
cm = pd.crosstab(y_valid, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)




