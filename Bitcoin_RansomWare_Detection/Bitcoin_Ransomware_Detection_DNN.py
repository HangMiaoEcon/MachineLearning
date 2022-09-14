# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 00:10:18 2022

@author: Hang
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 




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

Y = data.label
data_features = ['length', 'weight', 'count', 'looped','neighbors', 'income']
X = data[data_features]

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = utils.to_categorical(Y)
print(Y.shape)
print(X.shape)


from sklearn.preprocessing import MinMaxScaler
# Normalize features within range 0 (minimum) and 1 (maximum)
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X = pd.DataFrame(X)
# split full dataset into the training data set and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, test_size=0.1,
                                                                random_state=1)

# split training dataset further into the training data set and validating dataset
import numpy as np 
X_train, X_valid = np.split(X_train, [int(.8 *X_train.shape[0])])
y_train, y_valid = np.split(y_train, [int(.8 *y_train.shape[0])])

##########################################
# chcek GPU tensorflow is working
##########################################
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)
tf.config.list_physical_devices('GPU')


##############################################
# Set up Deep Learning Model
##############################################
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
# DNN Model
model = Sequential()
model.add(Dense(5, input_dim = 6, activation = 'relu')) # Rectified Linear Unit Activation Function
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(29, activation = 'softmax')) # Softmax for multi-class classification
# Compile model here
model.compile(loss = 'CategoricalCrossentropy', optimizer = 'adam', metrics = ['CategoricalAccuracy'])
# CategoricalAccuracy, SparseCategoricalAccuracy, Recall


history = model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=10,batch_size=32,verbose=1)
model.save('DDN_Model')
model.load('DDN_Model')

##############################################
# plot the traning and validation loss
##############################################

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()



##############################################
# Test dataset Prediction
##############################################
test_predict=model.predict(X_test)
y_pred = tf.argmax(test_predict, axis=1)
y_test = y_test.argmax(1)

from sklearn.metrics import precision_recall_fscore_support, auc, accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
# confusion matrix
#cm = confusion_matrix(y_test, test_predict)
# precision, recall, fscore, support
prfs = precision_recall_fscore_support(y_pred, y_test )
y_pred = le.inverse_transform(test_predict)











