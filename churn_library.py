'''
This module converts churn_notebook.ipynb into a modular code that
is easy to test and debug. It includes functions for:
1) importing the data
2) performing EDA
3) encoding cat data
4) performing feature engineering
5) training a model
6) generating classification report
7) plotting feature importance

Date: 2023-09-25
Author: Vadim Polovnikov
'''

# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import shap # building and explaining graphs
import joblib # saving scikit-learn models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

from pathlib import Path


def import_data(pth):
    '''
    Returns Pandas DataFrame.
    Input:
        - pth: (str) file path
    Output:
        - Pandas DF
    '''
    df = pd.read_csv(pth)
    return df


# importing the dataset
df = import_data('data/bank_data.csv')

# dropping extraneous column
df.drop(labels="Unnamed: 0", axis='columns', inplace=True)

# Adding 'Churn' col based on the Attrition_Flag value
df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
df.loc[df['Attrition_Flag'] != 'Existing Customer'].head() 


def perform_eda(df, col1='Churn', col2='Customer_Age',
                col3='Marital_Status', col4='Total_Trans_Ct'):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    # need to modularize this!
    plt.figure(figsize=(10,5))

    # place into constants.py
    IMAGES_PATH_RESULTS  = Path() / "images" / "results"
    IMAGES_PATH_EDA = Path() / "images" / "eda"
    IMAGES_PATH_RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES_PATH_EDA.mkdir(parents=True, exist_ok=True)

    # churn hist
    plt.hist(df[col1]) # col1
    plt.xlabel('Churn: 1 - yes, 0  - no')
    plt.ylabel('Number of customers')
    churn_hist_pth = IMAGES_PATH_EDA / 'churn_hist.png'
    plt.savefig(churn_hist_pth, format='png', dpi='figure')

    # churn_cx_age
    plt.hist(df[col2]) #col2
    plt.xlabel('Customers\' age')
    plt.ylabel('Number of Customers')
    churn_cx_age_pth = IMAGES_PATH_EDA / 'churn_cx_age.png'
    plt.savefig(churn_cx_age_pth, format='png', dpi='figure')

    # churn_marital_status
    plt.ylabel('Number of customers: normalized')
    df[col3].value_counts('normalize').plot(kind='bar') #col3
    churn_marital_status_pth = IMAGES_PATH_EDA / 'churn_marital_status.png'
    plt.savefig(churn_marital_status_pth, format='png', dpi='figure')

    # total_trans_ct
    sns.histplot(df[col4], stat='density', kde=True) #col4
    total_trans_ct_pth = IMAGES_PATH_EDA / 'total_trans_ct.png'
    plt.savefig(total_trans_ct_pth, format='png', dpi='figure')

    # corr_matrix
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    corr_matrix_pth = IMAGES_PATH_EDA / 'corr_matrix.png'
    plt.savefig(corr_matrix_pth, format='png', dpi='figure')


# performing eda
perform_eda(df)

y = df['Churn']
X = pd.DataFrame()

def encoder_helper(df, category, new_cat_name):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            category: column to base the cat-to-num transfomration of
            new_cat_name: new numerical feature name

    output:
            df: pandas dataframe with new columns for
    '''
    numeric_vals_lst = []
    category_groups = df.groupby(category).mean()['Churn']

    for val in df[category]:
        numeric_vals_lst.append(category_groups.loc[val])

    df[new_cat_name] = numeric_vals_lst  


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


'''if __name__ == "__main__":
    # importing the dataset
    df = import_data('data/bank_data.csv')

    # dropping extraneous column
    df.drop(labels="Unnamed: 0", axis='columns', inplace=True)

    # Adding 'Churn' col based on the Attrition_Flag value
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df.loc[df['Attrition_Flag'] != 'Existing Customer'].head() 
    
    # performing eda
    perform_eda(df)'''