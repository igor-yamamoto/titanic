## Library importation
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from func import data_investigation, hist_plot, get_subplot_dims, stack_plot, hist_stack_plot

from scipy.stats import pointbiserialr as pbr

## Function definition
def drop_feat(df, attr):
    for ft in attr:
        if ft in df.columns:
            return df.drop(attr, axis = 1)
        else:
            print('No such feature as ' + ft)

def plot_multi_hist(data1, data2, attr, label):
#    ax = ((ax0, ax1), (ax2, ax3))
    
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    
    nbins = 50
    ax0.hist([data1[attr[0]], data2[attr[0]]], nbins, label = label)
    ax1.hist([data1[attr[1]], data2[attr[1]]], nbins, label = label)
    ax2.hist([data1[attr[2]], data2[attr[2]]], nbins, label = label)
    ax3.hist([data1[attr[3]], data2[attr[3]]], nbins, label = label)
    plt.legend()
    plt.show()
    
    
def prb_calc(label, feat, cols_name):
    a, b = np.shape(tr_prepared)
    r_calc = np.zeros(8)
    r_calc = r_calc.transpose()
    for i in range(b):
        r = pbr(label, feat[:, i])
        r_calc[i] = r[0]
        
    r_calc = pd.DataFrame(r_calc).T
    r_calc.columns = cols_name
    return r_calc

def array_to_df(array, cols_name):
    df = pd.DataFrame(array)
    df.columns = cols_name
    return df

def cut_df(df, targets, bins):
    df_cut = pd.DataFrame(index = df.index, columns = targets)
    for i in range(len(targets)):
        df_cut[targets[i]] = pd.cut(df[targets[i]], bins[i])
    
    return df_cut

class CustImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        a = X.isnull().sum()
        b = a[a>0]
        c = b.index
        
        for i in c:
            type_var = X[i].dtype
            if (type_var == 'int64') or (type_var == 'float64'):
                X[i] = X[i].fillna(X[i].mean())
            elif (type_var == 'O'):
                X = X.dropna(subset = [i], axis = 0)
            
        return X
    
class CombinedAttr(BaseEstimator, TransformerMixin):
    def __init__(self, add_family_size = False):
        self.add_family_size = add_family_size
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        #class_id = X.columns.get_loc('Pclass')
        #age_id = X.columns.get_loc('Age')
        #fare_id = X.columns.get_loc('Fare')
        print(X)
        class_age = X['Pclass'] * X['Age']
        class_age.columns = ['Class*Age']
        class_fare = X['Pclass'] * X['Fare']
        class_fare.columns = ['Class*Fare']
        if self.add_family_size:
            family_size = X['SibSp'] + X['Parch'] + 1 
            family_size.columns = ['Family Size']
            X['Class*Age'] = class_age
            X['Class*Fare'] = class_fare
            X['Family Size'] = family_size
            return X
        else:
            X['Class*Age'] = class_age
            X['Class*Fare'] = class_fare            
            return X

def extract_feat_name(df):
    
    att = CombinedAttr()
    df = att.transform(df)
    
    a = df.dtypes
    num = pd.DataFrame()
    obj = pd.DataFrame()
    obj_encoded = pd.DataFrame()
    
    for i in range(len(a)):
        if (a[i] == 'int64') or (a[i] == 'float64'):
            num[a.index[i]] = a.index[i]
        elif (a[i] == 'O'):
            obj[a.index[i]] = a.index[i]
    
    for i in obj.columns:
        b = df[i].value_counts()
        for j in sorted(b.index):
            obj_encoded[j] = j
    
    return list(num.columns) + list(obj_encoded.columns)
        

## Dataset importation
test_path = '../datasets/test.csv'
train_path = '../datasets/train.csv'

test_set = pd.read_csv(test_path)
train_set = pd.read_csv(train_path)

test_set.head()
test_set.info()

train_set.head()
train_set.info()

## amount of null data

train_set.describe()
descr_survive = train_set[train_set['Survived'] == 1].describe()
descr_not_survive = train_set[train_set['Survived'] == 0].describe()

corr_matrix = train_set.corr()
corr_matrix['Survived'].sort_values(ascending=False)

data_investigation(train_set)

hist_stack_plot(train_set, ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'])

## extracting title
train_set['Title'] = train_set['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
train_set.groupby('Title')['Age'].mean()
train_set['Title'].value_counts()

## filling age by title
train_set['Age'].fillna(train_set.groupby('Title')['Age'].transform('mean'), inplace = True)

## FEAT ENGINEERING - checking for possible features that can be used instead of the presented ones
## Family Size
train_set['Family Size'] = train_set['SibSp'] + train_set['Parch'] + 1

train_set['Class*Fare'] = train_set['Pclass'] * train_set['Fare']

train_set['Class*Age'] = train_set['Pclass'] * train_set['Age']

corr_matrix = train_set.corr()
corr_matrix['Survived'].sort_values(ascending=False)

train_set[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = 'Parch')
train_set[['Class*Age', 'Survived']].groupby(['Class*Age'], as_index = False).mean().sort_values(by = 'Class*Age')

to_drop = ['Name', 'Ticket', 'Cabin']

train_set = drop_feat(train_set, to_drop)
train_set = train_set.dropna(subset = ['Sex', 'Embarked'])
test_set = drop_feat(test_set, to_drop)

attr = ['Survived', 'Pclass', 'Fare', 'Sex', 'Age', 'Embarked']

#plot_multi_hist(survived_df, not_survived_df, attr, ['Surv', 'No Surv'])

## Data preparation
## Test-Train set spliting
tr_set, ts_set = train_test_split(train_set[attr], test_size = 0.2, random_state = 42)

## Cleaning
tr_label = tr_set['Survived']
ts_label = ts_set['Survived']

tr_set = drop_feat(tr_set, ['Survived'])
ts_set = drop_feat(ts_set, ['Survived'])

tr_num = drop_feat(tr_set, ['Sex', 'Embarked'])
tr_cat = tr_set[['Sex','Embarked']]

ts_num = drop_feat(ts_set, ['Sex', 'Embarked'])
ts_cat = ts_set[['Sex','Embarked']]

## Pipelining
## numerical pipe
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'median')),
        ('std_scaler', StandardScaler()),
        ])

## categorical and numerical pipe (full pipe)
num_attr = list(tr_num)
cat_attr = ['Sex','Embarked']
#
#full_pipeline = ColumnTransformer([
#        ('num', num_pipeline, num_attr),
#        ('cat', OneHotEncoder(), cat_attr),
#        ])
attr_inpute = CustImputer()
tr_set = attr_inpute.transform(tr_set)
   
num_pipe = Pipeline([
#    ('imputer', SimpleImputer(strategy = 'mean')), 
#    ('imputer', CustImputer()), 
    ('custom_transform', CombinedAttr()),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ('num', num_pipe, num_attr),
    ('cat', OneHotEncoder(), cat_attr),
])

tr_prepared = full_pipeline.fit_transform(tr_set)
ts_prepared = full_pipeline.fit_transform(ts_set)


## point biserial r calculation
cols_name = ['PClass', 'Fare', 'Age', 'Female', 'Male', 'C', 'Q', 'S']
r_sur_calc = prb_calc(tr_label, tr_prepared, cols_name)

tr_prepared = array_to_df(tr_prepared, cols_name)
ts_prepared = array_to_df(ts_prepared, cols_name)

tr_prepared = drop_feat(tr_prepared, 'Q')
ts_prepared = drop_feat(ts_prepared, 'Q')

## DATA EXPORTATION
pd.DataFrame(tr_prepared).to_csv('datasets/treated/train_prepared.csv')
pd.DataFrame(tr_label).to_csv('datasets/treated/train_label.csv')

pd.DataFrame(ts_prepared).to_csv('datasets/treated/test_prepared.csv')
pd.DataFrame(ts_label).to_csv('datasets/treated/test_label.csv')