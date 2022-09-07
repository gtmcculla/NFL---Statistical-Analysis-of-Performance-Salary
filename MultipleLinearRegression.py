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

x = fulldf[['YDS', 'REC']].values.reshape(-1,2)
y = fulldf['Money']

x1 = x[:, 0]
y1 = x[:, 1]
z=y
x_pred = np.linspace(0, 2000, 49)
y_pred = np.linspace(0, 150, 49)

xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

ols = linear_model.LinearRegression()
model = ols.fit(x, y)
predicted = model.predict(model_viz)

r2 = model.score(x, y)

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(111, projection='3d')
#ax2 = fig.add_subplot(132, projection='3d')
#ax3 = fig.add_subplot(133, projection='3d')

axes = ax1

#for ax in axes:
ax1.plot(x1, y1, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
ax1.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
ax1.set_xlabel('Yards', fontsize=12)
ax1.set_ylabel('Recptions', fontsize=12)
ax1.set_zlabel('Money', fontsize=12)
ax1.locator_params(nbins=4, axis='x')
ax1.locator_params(nbins=5, axis='x')

ax1.view_init(elev=28, azim=120)
#ax2.view_init(elev=4, azim=114)
#ax3.view_init(elev=60, azim=165)

fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)

fig.tight_layout()

for ii in np.arange(0, 360, 1):
    ax1.view_init(elev=32, azim=ii)
    fig.savefig('C:/Users/tybru/OneDrive/Desktop/gif/gif_image%d.png' % ii)

plt.show()