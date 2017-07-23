import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0,8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

pd.options.mode.chained_assignment = None  # default='warn'

train = pd.read_csv("C:\\HousingData\\train.csv")
test = pd.read_csv("C:\\HousingData\\test.csv")

train.head()

print ('The train data has {0} rows and {1} columns'.format(train.shape[0],train.shape[1]))
print ('----------------------------')
print ('The test data has {0} rows and {1} columns'.format(test.shape[0],test.shape[1]))


#train.info()

#Check missing Values
train.columns[train.isnull().any()]

miss = train.isnull().sum()/len(train)

miss = miss[miss >0]
miss.sort_values(inplace = True)
#print(miss)

miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index

#plot the mssing value count

sns.set(style="whitegrid", color_codes = True)
sns.barplot(x='Name', y='count', data = miss)
plt.xticks(rotation=90)
#sns.plt.show()

#saleprice
sns.distplot(train['SalePrice'])

print("The skewness of sale price is {}".format(train['SalePrice'].skew()))

target = np.log(train['SalePrice'])
print('Skewness is', target.skew())
#sns.distplot(target)

numeric_data = train.select_dtypes(include=[np.number])
cat_data = train.select_dtypes(exclude = [np.number])
print ("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))
del numeric_data['Id']
corr = numeric_data.corr()
sns.heatmap(corr)
print (corr['SalePrice'].sort_values(ascending=False)[:15], '\n') #top 15 values
print ('----------------------')
print (corr['SalePrice'].sort_values(ascending=False)[-5:]) #last 5 values
train['OverallQual'].unique()
#let's check the mean price per quality and plot it.
pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
#pivot.sort
pivot.plot(kind='bar', color='red')
#GrLivArea variable
sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'])
cat_data.describe()
sp_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
sp_pivot.plot(kind='bar',color='red')
cat = [f for f in train.columns if train.dtypes[f] == 'object']
def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = cat
    pvals = []
    for c in cat:
           samples = []
           for cls in frame[c].unique():
                  s = frame[frame[c] == cls]['SalePrice'].values
                  samples.append(s)
           pval = stats.f_oneway(*samples)[1]
           pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

cat_data['SalePrice'] = train.SalePrice.values
k = anova(cat_data) 
k['disparity'] = np.log(1./k['pval'].values) 
sns.barplot(data=k, x = 'features', y='disparity') 
plt.xticks(rotation=90) 
#create numeric plots
num = [f for f in train.columns if train.dtypes[f] != 'object']
num.remove('Id')
nd = pd.melt(train, value_vars = num)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')
def boxplot(x,y,**kwargs):
            sns.boxplot(x=x,y=y)
            x = plt.xticks(rotation=90)

cat = [f for f in train.columns if train.dtypes[f] == 'object']

p = pd.melt(train, id_vars='SalePrice', value_vars=cat)
g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, 'value','SalePrice')