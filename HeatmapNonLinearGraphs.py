from unicodedata import name
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

fulldf = pd.read_csv('C:/Users/tybru/Downloads/receiving2.csv')
fulldf = fulldf.dropna()
fulldf = fulldf.astype({'YDS': int})
fulldf = fulldf[fulldf.AGE > 27 ]

heatmapdf = fulldf.drop(columns=['Nums', 'YDS.1', 'TD.1', 'Rank', 'LG', 'ROST', 'FPTS','G','FL', 'ATT', '20+'])
#sns.heatmap(heatmapdf.corr(), annot=True)

plt.scatter(fulldf.Money, fulldf.YDS, facecolors='None', edgecolors='k', alpha=.5)
sns.regplot(fulldf.Money, fulldf.YDS, ci=None, label='Linear', scatter=False, color='orange')
sns.regplot(fulldf.Money, fulldf.YDS, ci=None, label='Degree 2', order=2, scatter=False, color='lightblue')
sns.regplot(fulldf.Money, fulldf.YDS, ci=None, label='Degree 5', order=5, scatter=False, color='g')
plt.legend()

plt.show()