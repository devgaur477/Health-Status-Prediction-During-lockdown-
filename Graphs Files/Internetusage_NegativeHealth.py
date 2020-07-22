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
n  = ['-2 ' , '-3 ' , '-4 ' , '-5 ' , '-6 ' , '-7 ' , '-8 ' , '-9 ' , '-10' , '-11 ' , '-12 ']

test = df[df['Internet_R'].isin(options) & df['Age_R'].isin(age_d)& df['status health'].isin(n)]


x  = test.iloc[: , 13].values
y = test.iloc[: , -1].values


plt.hist(x)
plt.title('Frequence of Negative Health vs Internet Usage for 21-25 years old')
plt.xlabel('Internet Usage')
plt.ylabel('Frequence of Negative Mental')
plt.show()
