## Library importation
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, cross_validate
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, mean_squared_error


## Function definition
def resume_scores(arr_name, arr_scores, arr_CM, arr_score_name):
    print('Metrics for each model: \n')
    for i in range(len(arr_name)):
        a = arr_scores[i]
        print(' -' + arr_name[i] + ':')
        for j in arr_score_name:
            print(' --' + j + ': ' + str(a[j].mean()))
        print(' --Confusion matrix: \n' + str(arr_CM[i]) + '\n')
        print('-'*30 + '\n')
        

def submit_to_kaggle(submit = False):  
    if submit == True:
        import os
        comm = 'kaggle competitions submit -f ../datasets/submission/submission_titanic.csv -m "My submission" -q titanic'
        os.system(comm)
        
        
#------
        
path_treated = '../datasets/treated/'

test_set_prep = pd.read_csv(path_treated + 'test_prepared.csv')
train_set_prep = pd.read_csv(path_treated + 'train_prepared.csv')
train_label_prep = pd.read_csv(path_treated + 'train_label_prepared.csv', header = None)


scoring = ['accuracy', 'neg_mean_squared_error', 'precision', 'recall', 'f1']

log_reg = LogisticRegression(max_iter = 1000)
log_reg.fit(train_set_prep, train_label_prep)
pred_log = cross_val_predict(log_reg, train_set_prep, train_label_prep, cv = 10)

scores_log = cross_validate(log_reg, train_set_prep, train_label_prep, cv = 10, scoring = scoring)
CM_log = confusion_matrix(train_label_prep, pred_log)


forest_clf = RandomForestClassifier(random_state = 42)
forest_clf.fit(train_set_prep, train_label_prep)
pred_forest = cross_val_predict(forest_clf, train_set_prep, train_label_prep, cv = 3)

scores_forest = cross_validate(forest_clf, train_set_prep, train_label_prep, scoring = scoring, cv = 10)
CM_forest = confusion_matrix(train_label_prep, pred_forest)


names = ['Logistic Regression', 'Random Forest']
arr_scores = [scores_log, scores_forest]
arr_CM = [CM_log, CM_forest]
arr_score_name = ['test_accuracy', 'test_neg_mean_squared_error', 'test_precision', 'test_recall', 'test_f1']

resume_scores(names, arr_scores, arr_CM, arr_score_name)


pipe = Pipeline(steps=[('logistic', LogisticRegression(max_iter = 10000)),
])

C = np.logspace(-0.5, 1, 30)
penalty = ['l2']
solver = ['lbfgs', 'newton-cg', 'liblinear', 'sag']

param_grid = dict(logistic__C = C,
                  logistic__penalty = penalty,
                  logistic__solver = solver,
                  
)

clf_logistic = GridSearchCV(pipe, param_grid)
clf_logistic.fit(train_set_prep, train_label_prep)
clf_logistic.best_estimator_.get_params()

scores_log_opt = cross_validate(clf_logistic, train_set_prep, train_label_prep, scoring = scoring, cv = 3)
pred_log_opt = cross_val_predict(clf_logistic, train_set_prep, train_label_prep, cv = 3)
CM_log_opt = confusion_matrix(train_label_prep, pred_log_opt)


n_estimators = [10, 30, 50, 70]
max_depth = np.linspace(5, 30, 5)
min_samples_split = [0.01, 0.02575, 0.0415]
min_samples_leaf = np.linspace(1, 10, 4, dtype = int)

param_grid = dict(n_estimators = n_estimators,
                  max_depth = max_depth,
                  min_samples_split = min_samples_split,
                  min_samples_leaf = min_samples_leaf,
)

clf_forest = GridSearchCV(forest_clf, param_grid, cv = 3)
clf_forest.fit(train_set_prep, train_label_prep)
clf_forest.best_estimator_.get_params()

scores_forest_opt = cross_validate(clf_forest, train_set_prep, train_label_prep, scoring = scoring, cv = 3)
pred_forest_opt = cross_val_predict(clf_forest, train_set_prep, train_label_prep, cv = 3)
CM_forest_opt = confusion_matrix(train_label_prep, pred_forest_opt)


names = ['Logistic Regression', 'Logistic Regression (optimized)'
         , 'Random Forest', 'Random Forest (optimized)']
arr_scores = [scores_log, scores_log_opt, scores_forest, scores_forest_opt]
arr_CM = [CM_log, CM_log_opt, CM_forest, CM_forest_opt]

resume_scores(names, arr_scores, arr_CM, arr_score_name)

##-------Predicting with random forest-------##
test_pred = clf_forest.predict(test_set_prep)

out_df = pd.DataFrame({
    'PassengerId' : test_set_prep.PassengerId.astype(int), 
    'Survived' : test_pred
})

out_df.to_csv('../datasets/submission/submission_titanic.csv', index = False)
print("Your submission was successfully saved!")

submit_to_kaggle(submit = False)