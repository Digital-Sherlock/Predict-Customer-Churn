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
    Performs EDA on df and specified columns and saves figures to images folder.
    input:
            df: pandas dataframe
            col1: (str) df column (Churn - default)
            col2: (str) df column (Customer_Age - default)
            col3: (str) df column (Marital_Status - default)
            col4: (str) df column (Total_Trans_Ct - default)

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


def encoder_helper(df, category, new_cat_name):
    '''
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category.

    input:
            df: pandas dataframe
            category: column to base the cat-to-num transfomration on
            new_cat_name: new numerical feature name

    output:
            df: pandas dataframe with new columns for
    '''
    numeric_vals_lst = []
    category_groups = df.groupby(category).mean()['Churn']

    for val in df[category]:
        numeric_vals_lst.append(category_groups.loc[val])

    df[new_cat_name] = numeric_vals_lst
    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'                
    ]

    # encoding cat columns
    for cat in cat_columns:
        df = encoder_helper(df, cat, f'{cat}_Churn')

    # defining input and output variables
    X = pd.DataFrame()
    y = df['Churn']

    # Defining feature set for training data
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
        'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test
    


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