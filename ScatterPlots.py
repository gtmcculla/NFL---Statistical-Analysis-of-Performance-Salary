from unicodedata import name
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
 
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
 
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
 
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    
    y_pred = b_0 + b_1*x
    return y_pred


url = 'https://www.fantasypros.com/nfl/stats/wr.php'
html = urlopen(url)
statsPage = BeautifulSoup(html)


columnHeaders = statsPage.findAll('tr') [1]
columnHeaders = [i.getText() for i in columnHeaders.findAll('th')]


rows = statsPage.findAll('tr')[2:]

wrStats = []
for i in range(len(rows)):
       wrStats.append([col.getText() for col in rows[i].findAll('td')])
recievingData = pd.DataFrame(wrStats, columns=columnHeaders[0:])


url = 'https://overthecap.com/position/wide-receiver/'
html = urlopen(url)
statsPage = BeautifulSoup(html)

columnHeaders = statsPage.findAll('tr') [0]
columnHeaders = [i.getText() for i in columnHeaders.findAll('th')]

rows = statsPage.findAll('tr')[1:]

wrSalary = []
for i in range(len(rows)):
       wrSalary.append([col.getText() for col in rows[i].findAll('td')])

salaryData = pd.DataFrame(wrSalary, columns=columnHeaders[0:])

recievingData = recievingData.astype({'REC': int, 'TGT': int})
salaryData[salaryData.columns[5:6]] = salaryData[salaryData.columns[5:6]].replace('[\$,]', '', regex=True).astype(float)

fulldf = pd.read_csv('C:/Users/tybru/Downloads/receiving2.csv')
fulldf = fulldf.dropna()
fulldf = fulldf.astype({'YDS': int})
fulldf = fulldf[fulldf.AGE > 27 ]

x = fulldf['Money']
y1 = fulldf['REC']
y2 = fulldf['YDS']
y3 = fulldf['TD']
y4 = fulldf['Y/R']
y5 = fulldf['Catch%']
y6 = fulldf['FPTS/G']


#fig, ax = plt.subplots()

sns.set()
fig, axes = plt.subplots(3,2)

sns.scatterplot(data=fulldf, x='Money', y='REC', ax=axes[0,0])
sns.scatterplot(data=fulldf, x='Money', y='YDS', ax=axes[0,1])
sns.scatterplot(data=fulldf, x='Money', y='TD', ax=axes[1,0])
sns.scatterplot(data=fulldf, x='Money', y='Y/R', ax=axes[1,1])
sns.scatterplot(data=fulldf, x='Money', y='Catch%', ax=axes[2,0])
sns.scatterplot(data=fulldf, x='Money', y='FPTS/G', ax=axes[2,1])

yPred = estimate_coef(x,y1)
sns.lineplot(data=fulldf, x='Money', y=yPred, ax=axes[0,0])
yPred = estimate_coef(x,y2)
sns.lineplot(data=fulldf, x='Money', y=yPred, ax=axes[0,1])
yPred = estimate_coef(x,y3)
sns.lineplot(data=fulldf, x='Money', y=yPred, ax=axes[1,0])
yPred = estimate_coef(x,y4)
sns.lineplot(data=fulldf, x='Money', y=yPred, ax=axes[1,1])
yPred = estimate_coef(x,y5)
sns.lineplot(data=fulldf, x='Money', y=yPred, ax=axes[2,0])
yPred = estimate_coef(x,y6)
sns.lineplot(data=fulldf, x='Money', y=yPred, ax=axes[2,1])
#plt.xticks([20000000, 40000000, 60000000], ['20000000', '40000000', '60000000'])
plt.setp(axes, xticks=[20000000, 40000000, 60000000], xticklabels=['$20,000,000', '$40,000,000', '$60,000,000'])

contract = 20000000
lm = smf.ols('Money ~ TD + REC + YDS', fulldf).fit()
print(lm.summary())
lm = sm.OLS(y2,x).fit()
print(lm.summary())
lm = sm.OLS(y3,x).fit()


print(lm.summary())
lm = sm.OLS(y4,x).fit()
print(lm.summary())
lm = sm.OLS(y5,x).fit()
print(lm.summary())
lm = sm.OLS(y6,x).fit()
print(lm.summary())


plt.show()


