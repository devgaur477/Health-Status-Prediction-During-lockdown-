import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial import ConvexHull
import warnings; warnings.simplefilter('ignore')
import seaborn as sns


df = pd.read_csv('dataset.csv')

options = ['Very concerned ' , 'Somewhat concerned  ' , 'Not concerned  ']
age_d = [ '21-25 years old ']
age_n = ['16-20 years old ']

test = df[df['Concern_R'].isin(options) & df['Age_R'].isin(age_n)]


x  = test.iloc[: , 14].values
y = test.iloc[: , 25].values


print(y)
plt.hist(x)
plt.title('Stress due to their studies Age Group 16-20 Years Old')

plt.ylabel('Frequence')

plt.show()