#***** Toplu başucu olarak bakılabilecek bir kaynak *****#
#https://www.kaggle.com/vbmokin/eda-for-tabular-data-advanced-techniques

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats

# set warnings setting to ignore mode
import warnings
warnings.filterwarnings("ignore")

#set print format to float format for all numeric outputs to prevent like 'e+06' outputs
pd.set_option('display.float_format', lambda x: '%.3f' % x)


### EDA Guideline

#get data
data = pd.read_csv("sample_datasets/house_price.csv")
data2 = pd.read_csv("sample_datasets/titanic.csv")
data3 = pd.read_csv("sample_datasets/iris.csv")
data4 = pd.read_csv("sample_datasets/netflix_titles.csv")

#get first five record
data.head()

#get last five record
data.tail()

#get first n record
data.head(100)

#get last n record
data.tail(100)

#get shape of data
data.shape

#get label of each column
data.columns.values

#get data types of features
data.dtypes

#get data count of features
data.count()

#get features and their data types
#null_counts=True --> show features with null or non-null situations
data.info(null_counts=True)

#get summary descriptive statistics of columns with mean, std and etc.
data.describe()
data.describe(include='all')

#get number of duplicates entries
data.duplicated()
data.duplicated().sum()

#get unique values of categorical features
data.Foundation.unique()

#get counts of values in the categorical features
data.Foundation.value_counts()

#get number of missing values for each feature
data.isnull().sum()

#get percentage of missing values for each feature
#data.isnull() returns the same result in the next line below data.isna()
data.isna().mean().sort_values(ascending = False).head()

#create table of missing values
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#plot features with missing data
missing = data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace = True)

fig = plt.figure(figsize = (10,10))
sns.set(style = 'whitegrid')
ax = sns.barplot(x = missing.index.tolist(), y = missing, palette = 'hot_r')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

#get missing values on heatmap
#first parameter data's null situations of all cell
#cbar=False --> remove colorbar symbol on right of the graph
#cmap='viridis' --> heatmap colormap choice
plt.figure(figsize=(30, 30))
sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap='viridis')

#get numerical features
numerical_features = data.select_dtypes(exclude = ['object']).copy()
numerical_features.columns

#get categorical features
categorical_features = data.select_dtypes(include = ['object']).copy()
categorical_features.columns

#plot distributions of numerical features
fig = plt.figure(figsize = (20,25))
sns.set(style = 'whitegrid')
for i in range(len(numerical_features.columns)):
    fig.add_subplot((len(numerical_features.columns)/4)+1, 4, i+1)
    sns.distplot(numerical_features.iloc[:,i].dropna(), rug = True, hist = False, kde_kws = {'bw':0.1}, color = 'b')
    plt.xlabel(numerical_features.columns[i])
plt.tight_layout()

#plot statistics dispersion of numerical features
fig = plt.figure(figsize = (20,25))
sns.set(style = 'darkgrid')
for i in range(len(numerical_features.columns)):
    fig.add_subplot((len(numerical_features.columns)/4)+1, 4, i+1)
    sns.boxplot(y = numerical_features.iloc[:,i].dropna())
plt.tight_layout()

#plot bivariate analysis with all numerical feature vs target class
fig = plt.figure(figsize = (20,25))
sns.set(style = 'whitegrid')
for i in range(len(numerical_features.columns)):
    fig.add_subplot((len(numerical_features.columns)/4)+1, 4, i+1)
    sns.scatterplot(numerical_features.iloc[:,i].dropna(), data.SalePrice)
plt.tight_layout()

#get correlations of between all features
data.corr()

#show correlations with heatmap graph
#annot=True --> show correlation values on grid-cells
#cmap='viridis' --> heatmap colormap choice
plt.figure(figsize=(40, 40))
sns.heatmap(data.corr(), annot=True, cmap='viridis')

#plot corr only greater than threshold on heatmap
#data.corr() returns the same result as the next line below
correlation = data.select_dtypes(exclude = 'object').corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlation > 0.8, annot = True, square = True, cbar=False)

#plot corr only greater than threshold on heatmap with different style
plt.figure(figsize=(20, 20))
sns.heatmap(data.corr()[(data.corr() >= 0.5) | (data.corr() <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)

#get 10 most corr features on heatmap
k = 10
cols = data.corr().nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

#plot distributions of categorical features
fig = plt.figure(figsize = (20,50))
for i in range(len(categorical_features.columns)):
    fig.add_subplot((len(categorical_features.columns)/4)+1, 4, i+1)
    ax = sns.countplot(categorical_features.iloc[:,i].dropna())
    plt.xticks(rotation = 90)
plt.tight_layout()

#get summary descriptive statistics of categorical columns with count, freq and etc.
categorical_features.describe()
categorical_features.describe().T.freq.sort_values(ascending = False)

#get skewness
data.skew()
data['SalePrice'].skew()

#get kurtosis
data.kurt()
data['SalePrice'].kurt()

#plot bivariate analysis with all categorical feature vs target class
fig = plt.figure(figsize = (20,25))
sns.set(style = 'whitegrid')
for i in range(len(categorical_features.columns)):
    fig.add_subplot((len(categorical_features.columns)/4)+1, 4, i+1)
    ax = sns.boxplot(categorical_features.iloc[:,i].dropna(), data.SalePrice)
    ax.axis(ymin=0, ymax=800000)
plt.tight_layout()

#draw histogram with standard norm graph on histogram
sns.distplot(data['SalePrice'], fit=norm)

#draw check normal distribution/probability plot
stats.probplot(data['SalePrice'], plot=plt)

#create table of groupby feature values vs target class
data[["OverallQual", "SalePrice"]].groupby(['OverallQual'], as_index=False).mean().sort_values(by='SalePrice', ascending=True)

#draw distribution of features values approach target class's different values
g = sns.FacetGrid(data2, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(data2, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

#create crosstab between features
pd.crosstab(pd.cut(data2['Age'], bins=8), data2['Sex'])
pd.crosstab(pd.cut(data2['Age'], bins=8), [data2['Sex'], data2['Survived']])

#feature histogram by target class
plt.hist(x = [data2[data2['Survived']==1]['Age'], data2[data2['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

#compare feature values with another variable
fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))
#how does embarked factor with sex & survival compare
sns.pointplot(x="Embarked", y="Survived", hue="Sex", data=data2,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)
#how does class factor with sex & survival compare
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data2,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)

#pair plots of entire dataset
pp = sns.pairplot(data2, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', 
                  diag_kws=dict(shade=True), plot_kws=dict(s=10))
pp.set(xticklabels=[])

#find outliers with IQR
Q1 = data2.quantile(0.25)
Q3 = data2.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
data2_without_outlier = data2[~((data2 < (Q1 - 1.5 * IQR)) | (data2 > (Q3 + 1.5 * IQR))).any(axis=1)]
data2_without_outlier.shape

#find outliers with z-score
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(data2.select_dtypes(exclude = ['object'])))
print(z)
threshold = 3
print(np.where(z > 3)) #first array is row and second array is col number

#find outliers with ISO
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.1)
values = iso.fit_predict(data2.select_dtypes(exclude = ['object']))

#word cloud
from wordcloud import WordCloud
plt.figure(figsize=(40,20))
wordcloud = WordCloud(
                      background_color='Black',
                      width=1920,
                      height=1080
                                ).generate(" ".join(data4.description))
plt.imshow(wordcloud)
plt.axis('off')
