## Library importation
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt


## Function definition
def data_investigation(data):
    print('Number of instances : ' + str(data.shape[0]))
    print('Number of variables : ' + str(data.shape[1]))
    print('-'*20)
    print('Attributes, data type and ratio of unique instances por total non-null:')
    for i in range(data.shape[1]):
        print('\t - ' + str(data.columns[i]) + ', ' + str(data.dtypes[i]) 
              + ', ' + str(len(data[data.columns[i]].value_counts())) + '/' + 
             str(sum(data[data.columns[i]].value_counts())))
    
    print('-'*20)
    print('Attributes that have missing values: ')
    sum_missing_val = data.isnull().sum()
    print(sum_missing_val[sum_missing_val>0])
    print('-'*20)
    print('Pictorial representation of missing values:')
    plt.figure(figsize=(10,8))
    sns.heatmap(data.isnull(), yticklabels = False, cmap = 'gray')
    plt.show()
    
def drop_feat(df, attr):
    for ft in attr:
        if ft in df.columns:
            return df.drop(attr, axis = 1)
        else:
            print('No such feature as ' + ft)

def corr_matrix_plot(corr_matrix):
    plt.figure(figsize=(10,8))
    
    ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
        horizontalalignment='right');
                       
def hist_stack_plot(df, targets):
    L = len(targets)
    arrang = np.arange(2)
    labels_stack = ['Survived', 'Not survived']
    
    plt.figure(figsize = (12, 12))
    for i in range(len(targets)):
        plt.subplot(2, L, i+1)
        sns.countplot(x = targets[i], data = df)
        plt.ylabel(targets[i])
        plt.xlabel('')
        plt.xticks(rotation=60)
        
    for i in range(len(targets)):
        plt.subplot(2, L, L+i+1)
        sur = df[df['Survived'] == 1][targets[i]].value_counts()
#        sur = sur.values.reshape((len(sur), 1))
        sur_n = np.array(sur)
        not_sur = df[df['Survived'] == 0][targets[i]].value_counts()
#        not_sur = not_sur.values.reshape((len(not_sur), 1))
        not_sur_n = np.array(not_sur)
        
        if (len(sur) <= len(not_sur)):
            lim = len(sur)
        else:
            lim = len(not_sur)
        
        for j in range(lim):
            a = sur_n[j]
            b = not_sur_n[j]
            
            c = (a, b)
            plt.bar(arrang, c, label = sur.index[j])
            
        plt.xticks(arrang, labels_stack)
        plt.ylabel(targets[i])
        plt.legend(loc = 'upper left')
    
    plt.tight_layout()
        
    plt.show()
    
def hist_plot(df, target):
    J, L = get_subplot_dims(target)
    
    plt.figure(figsize = (12, 12))
    for i in range(len(target)):
        plt.subplot(J, L, i+1)
        sns.countplot(x = target[i], data = df)
        plt.xticks(rotation=60)
        
    plt.tight_layout()
    
    plt.show()
    
def get_subplot_dims(targets):
    a = (int(len(targets)/3))*3
    
    if a == len(targets):
        j = int(a/3)
    elif a < len(targets):
        j = int(a/3) + 1
    return(j, 3)    
    
def cut_df(df, targets, bins):
    df_cut = pd.DataFrame(index = df.index, columns = targets)
    df_cut['Survived'] = df['Survived']
    for i in range(len(targets)):
        df_cut[targets[i]] = pd.cut(df[targets[i]], bins[i])
    
    return df_cut

def extract_titles(dataset):
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)

def insert_age_title_based(df):
    df['Age'].fillna(df.groupby('Title')['Age'].transform('mean'), inplace = True)
    
def simplify_titles(df):
    df['Title'] = df['Title'].replace([
        'Lady', 'Don', 'Dona', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer'
        ], 'Others')
    
def download_data(down = False):
    if down == True:
        import os
        comm = 'kaggle competitions download -c titanic -p ../datasets && unzip ../datasets/titanic.zip -d ../datasets'
        os.system(comm)
        

#------
        
download_data(down = True)

test_path = '../datasets/test.csv'
train_path = '../datasets/train.csv'

test_set = pd.read_csv(test_path)
train_set = pd.read_csv(train_path)


ts_head = test_set.head()
ts_info = test_set.info()
print(ts_head)
print(ts_info)

tr_head = train_set.head()
tr_info = train_set.info()
tr_descr = train_set.describe()
print(tr_head)
print(tr_info)
print(tr_descr)


data_investigation(train_set)
data_investigation(test_set)


train_set = drop_feat(train_set, ['Cabin'])
test_set = drop_feat(test_set, ['Cabin'])


corr_matrix = train_set.corr()
corr_matrix    
corr_matrix_plot(corr_matrix)


attr = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
hist_stack_plot(train_set, attr)
hist_plot(test_set, attr)

attr = ['Age', 'Fare', 'PassengerId']
bins = [5, 5, 5]
train_set_cut = cut_df(train_set, attr, bins)
hist_stack_plot(train_set_cut, attr)



extract_titles(train_set)
extract_titles(test_set)
train_set['Title'].value_counts()
train_set.groupby('Title')['Age'].describe()

insert_age_title_based(train_set)
insert_age_title_based(test_set)

train_set['Age'].isnull().sum(), test_set['Age'].isnull().sum()


hist_stack_plot(train_set, ['Title'])

simplify_titles(train_set)
simplify_titles(test_set)

hist_stack_plot(train_set, ['Title'])



possib_attr = ['Survived', 'Family Size', 'Class*Age', 'Class*Fare']
possib_feat = pd.DataFrame(index = train_set.index, columns = possib_attr)
possib_feat['Survived'] = train_set['Survived']
possib_feat.info()

possib_feat['Family Size'] = train_set['SibSp'] + train_set['Parch'] + 1
possib_feat['Class*Age'] = train_set['Pclass'] * train_set['Age']
possib_feat['Class*Fare'] = train_set['Pclass'] * train_set['Fare']

possib_corr_matrix = possib_feat.corr()
possib_corr_matrix
corr_matrix_plot(possib_corr_matrix)