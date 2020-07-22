import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial import ConvexHull
import warnings; warnings.simplefilter('ignore')
import seaborn as sns


df = pd.read_csv('dataset.csv')

options = ['More than 8 hours per day ' , '4-5 hours per day ' , '2-3 hours per day ' , '0-1 hours per day ']
age_d = [ '21-25 years old ']
age_n = ['16-20 years old ']

test = df[df['Internet_R'].isin(options) & df['Age_R'].isin(age_d)]


x  = test.iloc[: , 13].values
y = test.iloc[: , 25].values


print(y)
plt.hist(x)
plt.title('Internet Usage of Age Group 21-25 Years Old')
plt.xlabel('Time in Hours')
plt.ylabel('Frequence')

plt.show()