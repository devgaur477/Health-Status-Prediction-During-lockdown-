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
print('Please answer few questions.')
print('')
a = input('Are you addicted to Smoking? If yes, How much? Low Level = 0 , Medium Level = -1 , Hight Level = -2. And if No, then 0 :')
b = input('Are you addicted to Alochol? If yes, How much? Low Level = 0 , Medium Level = -1 , Hight Level = -2. And if No, then 0: ')
c = input('Are you addicted to Gaming? If yes, How much? Low Level = 0 , Medium Level = -1 , Hight Level = -2. And if No, then 0')
d = input('Are you addicted to Social Media? If yes, How much? Low Level = 0 , Medium Level = -1 , Hight Level = -2. And if No, then 0:')
e = input('Do you feel an undue pressure of not being productive?. If Yes then -1 , if No then 0 :')
print('Please Select one of the followings : ')
print('Enjoying at home and spending more time with my family = 1')
print('Confused =0 ')
print('Enjoyed for few days, now it is getting on my nerves. =-1')
print('Not enjoying at all. Have started to get irritated very easily = -2')
f = input()
print('Do you feel stressed?')
print('If Yes then please enter -2')
print('If no then please enter 0')
g = input()
print('Please Select one of the followings')
print('It will take sometime for me. But I will adapt myself.Then Enter 0')
print('I have adapted myself accordingly.(I am Taking Precautions and in a good mental state. Then please enter 2')
print('It will be very difficult for me to adapt to this situation. But I will try my best. Then please enter -1')
print('I don''t want to adapt myself. I am very tensed at the moment and I want everything to return to normal as soon as possible. Then please enter -2')
h = input()
print('Do you feel Isolated?')
print('If Yes then please enter -1')
print('If No then please enter 0')
print('If sometimes then please enter 0')
i = input()
print('Are you concerned about your studies?')
print('Very concerned, then enter -2')
print('Somewhat concerned then enter  0 ')
print('Not concerned then enter 0')
j = input()
print('Do you fear COVID 19 ? If Yes then enter -1 and if No then enter 0')
k = input()
print('Are you employed? If Yes then enter 0 and if No then enter -1')
l = input()


print('answer is ')

health = ann.predict(sc.transform([[a,b,c,d,e,f,g,h,i,j,k,l]]))

if health > 0.5:
    print("Hey! You don't have to worry =D. You are Absolutely Fine!")
else:
    print("Ah! :( Negative Health")


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
cm = confusion_matrix(y_test_classes, y_pred_classes)
print(cm)









