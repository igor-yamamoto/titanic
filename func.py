import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

def hist_plot(df, target):
    J, L = get_subplot_dims(target)
    
    plt.figure(figsize = (12, 12))
    for i in range(len(target)):
        plt.subplot(J, L, i+1)
        sns.countplot(x = target[i], data = df)
        
    plt.show()
    
def stack_plot(df, targets):
    sur = df[df['Survived'] == 1][targets].value_counts()
    not_sur = df[df['Survived'] == 0][targets].value_counts()
    
    df1 = pd.DataFrame([sur, not_sur])
    df1.index = ['Survived', 'Not survived']
    
    df1.plot(kind = 'bar', stacked = True, figsize = (10, 5))
    
def hist_stack_plot(df, targets):
    L = len(targets)
    arrang = np.arange(2)
    labels_stack = ['Survived', 'Not survived']
    
    plt.figure(figsize = (12, 12))
    for i in range(len(targets)):
        plt.subplot(2, L, i+1)
        sns.countplot(x = targets[i], data = df)
        
    for i in range(len(targets)):
        plt.subplot(2, L, L+i+1)
        sur = df[df['Survived'] == 1][targets[i]].value_counts()
#        sur = sur.values.reshape((len(sur), 1))
        sur_n = np.array(sur)
        not_sur = df[df['Survived'] == 0][targets[i]].value_counts()
#        not_sur = not_sur.values.reshape((len(not_sur), 1))
        not_sur_n = np.array(not_sur)
        
        print(sur)
        print(not_sur)
        
        for j in range(len(sur)):
            a = sur_n[j]
            b = not_sur_n[j]
            
            c = (a, b)
            plt.bar(arrang, c, label = sur.index[j])
            
        plt.xticks(arrang, labels_stack)
        plt.ylabel(targets[i])
        plt.legend()
        
    plt.show()


def data_investigation(data):
    print('Number of instances :' + str(data.shape[0]))
    print('Number of variables :' + str(data.shape[1]))
    print('-'*20)
    print('Attributes and data type:')
    for i in range(data.shape[1]):
        print('\t - ' + str(data.columns[i]) + ', ' + str(data.dtypes[i]))
    
    print('-'*20)
    print('Attributes that have missing values: ')
    sum_missing_val = data.isnull().sum()
    print(sum_missing_val[sum_missing_val>0])
    print('-'*20)
    print('Pictorial representation:')
    plt.figure(figsize=(8,6))
    sns.heatmap(data.isnull(), yticklabels = False, cmap = 'gray')
    plt.show()
    
def get_subplot_dims(targets):
    a = (len(targets)/3)*3
    
    if a == len(targets):
        j = a/3
    elif a < len(targets):
        j = a/3 + 1
    return(j, 3)
    

    

    