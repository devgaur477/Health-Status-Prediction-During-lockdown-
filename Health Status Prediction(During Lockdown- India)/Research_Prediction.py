#!/usr/bin/env python
# coding: utf-8

# In[2]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.__version__


# In[3]:


dataset = pd.read_csv('datafinal.csv')

x = dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values


# In[20]:


result =0
value_stat = []




for i in range(0 , 276):
    if(y[i] > 0):
        result = 1
    elif y[i] < 0:
            result = 0
    else:
        result =0.5
    value_stat.append(result)
        

final = np.array([value_stat])

status = final.T

y = status


# In[21]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[22]:



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[23]:


ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=14 , activation = 'relu'))

ann.add(tf.keras.layers.Dense(units=14 , activation = 'relu'))

ann.add(tf.keras.layers.Dense(units=1 , activation='relu'))

ann.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics= ['accuracy'])

ann.fit(X_train , y_train ,  epochs = 100)


# In[24]:


print('answer is ')

health = ann.predict(sc.transform([[0,0,0,0,0,1,0,2,0,-2,-1,-1]]))

if health > 0.5:
    print("Positive")
else:
    print("Negative")


# In[9]:


y_pred = ann.predict(X_test)

np.set_printoptions(precision = 2)

from sklearn.metrics import confusion_matrix
cutoff = 0.5
y_pred_classes = np.zeros_like(y_pred)
y_pred_classes[y_pred > cutoff] = 1
y_test_classes = np.zeros_like(y_pred)
y_test_classes[y_test > cutoff] = 1
cm = confusion_matrix(y_test_classes, y_pred_classes)
print(cm)









