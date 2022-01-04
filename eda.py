import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# set warnings setting to ignore mode
import warnings
warnings.filterwarnings("ignore")

#set print format to float format for all numeric outputs to prevent like 'e+06' outputs
pd.set_option('display.float_format', lambda x: '%.3f' % x)


### EDA Guideline

#get data
data = pd.read_csv("sample_datasets/house_price.csv")

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

#get features and their data types
#null_counts=True --> show features with null or non-null situations
data.info(null_counts=True)

#get summary descriptive statistics of columns with mean, std and etc.
data.describe()

#get number of missing values for each feature
data.isnull().sum()

#get percentage of missing values for each feature
#data.isnull() returns the same result in the next line below data.isna()
data.isna().mean().sort_values(ascending = False).head()

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

#plot distributions of features
fig = plt.figure(figsize = (20,25))
sns.set(style = 'whitegrid')
for i in range(len(numerical_features.columns)):
    fig.add_subplot((len(numerical_features.columns)/4)+1, 4, i+1)
    sns.distplot(numerical_features.iloc[:,i].dropna(), rug = True, hist = False, kde_kws = {'bw':0.1}, color = 'b')
    plt.xlabel(numerical_features.columns[i])
plt.tight_layout()

#plot statistics dispersion of features
fig = plt.figure(figsize = (20,25))
sns.set(style = 'darkgrid')
for i in range(len(numerical_features.columns)):
    fig.add_subplot((len(numerical_features.columns)/4)+1, 4, i+1)
    sns.boxplot(y = numerical_features.iloc[:,i].dropna())
plt.tight_layout()

#plot bivariate analysis with all feature vs target class
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

