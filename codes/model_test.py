## Library importation
import numpy as np

import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from func import data_investigation

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier


## FUNCTION DEFINITION
def rmse(predicted_vals_tr, cross_val_score, label):
    val = np.sqrt(mean_squared_error(predicted_vals_tr, label))
    val1 = np.sqrt(-cross_val_score)
    display_scores(val, val1)

def display_scores(score_train, scores_cross):
    print('Score over trained values:', score_train)
    mean = scores_cross.mean()
    print('Mean over cross validation scores:', mean)
#    print('Standard deviation:', scores.std())
    if score_train < mean:
        print('Model may be overfit')
    if score_train >= mean:
        print('Model may not be overfit')
        
def cat_metrics(label, pred_value):
    acc = accuracy_score(label, pred_value)
    conf_ma = confusion_matrix(label, pred_value)
    ps = precision_score(label, pred_value)
    rs = recall_score(label, pred_value)
    f1 = f1_score(label, pred_value)
    return (acc, conf_ma, ps, rs, f1)

## TREATED DATA IMPORTATION
path_ts = 'datasets/treated/'
path_tr = 'datasets/treated/'

ts_feat = pd.read_csv(path_ts + 'test_prepared.csv')
ts_label = pd.read_csv(path_ts + 'test_label.csv')
ts_label = ts_label['Survived']

tr_feat = pd.read_csv(path_tr + 'train_prepared.csv')
tr_label = pd.read_csv(path_tr + 'train_label.csv')
tr_label = tr_label['Survived']

ts_label_bool = np.array(ts_label, dtype=bool)
tr_label_bool = np.array(tr_label, dtype=bool)

## MODEL TESTING - Regression methods
## linear regression
lin_reg = LinearRegression()
lin_reg.fit(tr_feat, tr_label)
pred_lin_tr = lin_reg.predict(tr_feat)

scores_lin = cross_val_score(lin_reg, tr_feat, tr_label, scoring = 'neg_mean_squared_error', cv = 10)

rmse(pred_lin_tr, scores_lin, tr_label)

## decision tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(tr_feat, tr_label)
pred_tree_tr = tree_reg.predict(tr_feat)

scores_tree = cross_val_score(tree_reg, tr_feat, tr_label, scoring = 'neg_mean_squared_error', cv = 10)

rmse(pred_tree_tr, scores_tree, tr_label)

## random forest
forest_reg = RandomForestRegressor()
forest_reg.fit(tr_feat, tr_label)
pred_forest_tr = forest_reg.predict(tr_feat)

scores_forest = cross_val_score(forest_reg, tr_feat, tr_label, scoring = 'neg_mean_squared_error', cv = 10)

rmse(pred_forest_tr, scores_forest, tr_label)

## logistic regression
log_reg = LogisticRegression()
log_reg.fit(tr_feat, tr_label)
pred_log_tr = log_reg.predict(tr_feat)

scores_log = cross_val_score(log_reg, tr_feat, tr_label, scoring = 'neg_mean_squared_error', cv = 10)

rmse(pred_log_tr, scores_log, tr_label)


## MODEL TESTING - Classification methods
## Stochrastic Gradient Descent (SGD) Classifier
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(tr_feat, tr_label_bool)
pred_sgd = cross_val_predict(sgd_clf, tr_feat, tr_label_bool, cv = 3)

acc_sgd, conf_ma_sgd, ps_sgd, rs_sgd, f1_sgd = cat_metrics(tr_label, pred_sgd)

## Random Forest Classifier
forest_clf = RandomForestClassifier(random_state = 42)
pred_forest = cross_val_predict(forest_clf, tr_feat, tr_label_bool.reshape(-1, 1), cv = 3)

acc_forest, conf_ma_forest, ps_forest, rs_forest, f1_forest = cat_metrics(tr_label, pred_forest)