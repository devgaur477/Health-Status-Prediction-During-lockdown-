# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 00:02:07 2020

@author: DEV
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial import ConvexHull
import warnings; warnings.simplefilter('ignore')
import seaborn as sns


df = pd.read_csv('dataset.csv')

# X = df.iloc[: , 25].values
# y = df.iloc[: , -1].values
# z = df.iloc[: , 7].values
# age = []
# social_media = []

# c_type = 10

# for i in range(0,256):
#     age.append(X[i])

# for i in range(0,256):
#     social_media.append(z[i])

# combine = np.vstack((age , social_media)).T

options = ['High Level ' , 'Low Level ' , 'Medium Level ' , 'No ']
age_d = ['16-20 years old ']
status = ['-4']

result = df[df['Social Media_R'].isin(options)&df['Age_R'].isin(age_d)]

test = df[df['Social Media_R'].isin(options) & df['Age_R'].isin(age_d)]


x = test.iloc[: , 25].values
y = test.iloc[:, 7].values
# final_age = result.iloc[: , 25].values
# final_socialMedia = result.iloc[: , 7].values

plt.hist(y)
plt.title('The Usage of Social Media by the age group of 16-20 years old')
plt.ylabel('Frequence')
plt.xlabel('The level of usage')
plt.show()






