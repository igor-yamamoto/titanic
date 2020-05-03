## Library importation
import numpy as np

import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from scipy.stats import pointbiserialr as pbr

import math

## Function definition
def drop_feat(dataframe, array):
    return dataframe.drop(array, axis = 1)

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
    
def display_scores(scores):
    print('Scores:', scores)
    print('Mean', scores.mean())
    print('Standard deviation:', scores.std())
    
def prb_calc(label, feat, cols_name):
    a, b = np.shape(tr_prepared)
    r_calc = np.zeros(8)
    r_calc = r_calc.transpose()
    for i in range(b):
        r = pbr(label, feat[:, i])
        r_calc[i] = r[0]
    return r_calc

## Dataset importation
test_path = 'datasets/test.csv'
train_path = 'datasets/train.csv'

test_set = pd.read_csv(test_path)
train_set = pd.read_csv(train_path)

test_set.head()
test_set.info()

train_set.head()
train_set.info()

to_drop = ['Name', 'Ticket', 'Cabin']

train_set = drop_feat(train_set, to_drop)
train_set = train_set.dropna(subset = ['Sex', 'Embarked'])
test_set = drop_feat(test_set, to_drop)

corr_matrix = train_set.corr()
corr_matrix['Survived'].sort_values(ascending=False)


attr = ['Survived', 'Pclass', 'Fare', 'Age', 'Sex', 'Embarked']

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

full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attr),
        ('cat', OneHotEncoder(), cat_attr),
        ])

tr_prepared = full_pipeline.fit_transform(tr_set)
ts_prepared = full_pipeline.fit_transform(ts_set)


## point biserial r calculation
cols_name = np.array(['PClass', 'Fare', 'Age', 'Female', 'Male', 'C', 'Q', 'S'])
r_sur_calc = prb_calc(tr_label, tr_prepared, cols_name)
r_sur_calc

## Model testing
## linear regression
lin_reg = LinearRegression()
lin_reg.fit(tr_prepared, tr_label)
pred_lin = lin_reg.predict(tr_prepared)

lin_mse = mean_squared_error(pred_lin, tr_label)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

scores_lin = cross_val_score(lin_reg, tr_prepared, tr_label, scoring = 'neg_mean_squared_error', cv = 10)
lin_rmse_scores = np.sqrt(-scores_lin)
display_scores(lin_rmse_scores)

## decision tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(tr_prepared, tr_label)
pred_tree = tree_reg.predict(tr_prepared)

tree_mse = mean_squared_error(pred_tree, tr_label)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

scores_tree = cross_val_score(tree_reg, tr_prepared, tr_label, scoring = 'neg_mean_squared_error', cv = 10)
tree_rmse_scores = np.sqrt(-scores_tree)
display_scores(tree_rmse_scores)

## random forest
forest_reg = RandomForestRegressor()
forest_reg.fit(tr_prepared, tr_label)
pred_forest = forest_reg.predict(tr_prepared)

forest_mse = mean_squared_error(pred_forest, tr_label)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

scores_forest = cross_val_score(forest_reg, tr_prepared, tr_label, scoring = 'neg_mean_squared_error', cv = 10)
forest_rmse_scores = np.sqrt(-scores_forest)
display_scores(forest_rmse_scores)

## logistic regression
log_reg = LogisticRegression()
log_reg.fit(tr_prepared, tr_label)
pred_log = log_reg.predict(tr_prepared)

log_mse = mean_squared_error(pred_log, tr_label)
log_rmse = np.sqrt(log_mse)
log_rmse

scores_log = cross_val_score(log_reg, tr_prepared, tr_label, scoring = 'neg_mean_squared_error', cv = 10)
log_rmse_scores = np.sqrt(-scores_log)
display_scores(log_rmse_scores)