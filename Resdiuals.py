from unicodedata import name
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

fulldf = pd.read_csv('C:/Users/tybru/Downloads/receiving2.csv')
fulldf = fulldf.dropna()
fulldf = fulldf.astype({'YDS': int})
fulldf = fulldf[fulldf.AGE > 27 ]

sns.residplot(x='Money', y='REC', data=fulldf, lowess=True)
plt.show()