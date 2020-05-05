## Library importation
import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from data_analysis_2 import drop_feat, insert_age_title_based, extract_titles, simplify_titles, data_investigation


## Function definition
class CombinedAttr(BaseEstimator, TransformerMixin):
    def __init__(self, add_family_size = False):
        self.add_family_size = add_family_size
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
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
    
def transform_full(inp, contain_id = False, label = False):
    attr_impute = CustImputer()
    imputer_out = attr_impute.transform(inp)
    
    if label == True:
        label_out = imputer_out['Survived']
        imputer_out = drop_feat(imputer_out, ['Survived'])
    
    if contain_id == True:
        passenger_id = imputer_out['PassengerId'].values
        
        passenger_id = passenger_id.reshape(-1, 1)
        
        out = drop_feat(imputer_out, ['PassengerId'])

        out = full_pipe.fit_transform(imputer_out)
        
        array = np.concatenate((passenger_id, out), axis = 1)
        
    else:
        array = full_pipe.fit_transform(imputer_out)
        
    if label == True:
        return array, label_out
    else:
        return array

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

def general_transformation(df, train = True):
    df = drop_feat(df, ['Cabin'])
    
    extract_titles(df)
    insert_age_title_based(df)
    simplify_titles(df)
    
    df = drop_feat(df, ['Name', 'Ticket'])
    
    if train == True :
        train_df = drop_feat(df, ['Survived'])
        
        cols_name = extract_feat_name(train_df)
        
        transformed, train_label = transform_full(df, contain_id = True, label = True)
        
        transformed = pd.DataFrame(transformed)
        transformed.columns = cols_name

        return transformed, train_label
    else: 
        cols_name = extract_feat_name(df)
        
        transformed = transform_full(df, contain_id = True)

        transformed = pd.DataFrame(transformed)
        transformed.columns = cols_name

        return transformed

#------
        
test_path = '../datasets/test.csv'
train_path = '../datasets/train.csv'

test_set = pd.read_csv(test_path)
train_set = pd.read_csv(train_path)


attr_num = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
attr_cat = ['Sex', 'Embarked', 'Title']
attr_label = ['Survived']

num_pipe = Pipeline([
    ('custom_transform', CombinedAttr()),
    ('std_scaler', StandardScaler()),
])

full_pipe = ColumnTransformer([
    ('num', num_pipe, attr_num),
    ('cat', OneHotEncoder(), attr_cat),
])
    
    
train_set_prep, train_label_prep = general_transformation(train_set)
test_set_prep = general_transformation(test_set, train = False)

data_investigation(train_set_prep)
data_investigation(test_set_prep)


train_set_prep.to_csv('../datasets/treated/train_prepared.csv', index = False)
train_label_prep.to_csv('../datasets/treated/train_label_prepared.csv', index = False)
test_set_prep.to_csv('../datasets/treated/test_prepared.csv', index = False)