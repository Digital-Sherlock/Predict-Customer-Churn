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
    IMAGES_PATH_EDA = Path() / "images" / "eda"
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
    plt.close()


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


# identifying cat columns
cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'                
        ]

# defining input and output variables
X = pd.DataFrame()
y = df['Churn']


# defining input and output variables
X = pd.DataFrame()
y = df['Churn']


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
    # encoding cat columns
    for cat in cat_columns:
        df = encoder_helper(df, cat, f'{cat}_Churn')

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


# feature engneering and cat encoding
X_train, X_test, y_train, y_test = perform_feature_engineering(df)


def classification_report_image(model_name, y_train, y_test,
                                y_train_preds, y_test_preds):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions
            y_test_preds: test predictions

    output:
             None
    '''
    IMAGES_PATH_RESULTS  = Path() / "images" / "results"
    IMAGES_PATH_RESULTS.mkdir(parents=True, exist_ok=True)

    # scores
    plt.rc('figure', figsize=(10, 7))
    plt.text(0.01, 1.08, str(f'{model_name} Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(f'{model_name} Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    class_report = IMAGES_PATH_RESULTS / f'class_report_{model_name}.png'
    plt.savefig(class_report, format='png', dpi='figure')
    plt.close()


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
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,7))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    feature_imp_plt = output_pth / 'feature_imp_plt.png'
    plt.savefig(feature_imp_plt, format='png', dpi='figure');


def plot_roc_curves(model1, model2, X_test, y_test,
                    output_pth):
    '''
    Plots ROC curve based on models' performance
    on a test set.
    Input:
        - model: model to plot the ROC curve for
        - X_test: test dataset inputs
        - y_test: test dataset labels
    Output:
        - none
    '''
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    model1_roc_curve = plot_roc_curve(model1, X_test, y_test, ax=ax, alpha=0.8)
    model2_roc_curve = plot_roc_curve(model2, X_test, y_test, ax=ax, alpha=0.8)
    roc_curve = output_pth / 'ROC_Curve.png'
    plt.savefig(roc_curve, format='png', dpi='figure')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    The function trains models and saves the best ones.
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    IMAGES_PATH_RESULTS  = Path() / "images" / "results"
    IMAGES_PATH_RESULTS.mkdir(parents=True, exist_ok=True)

    # defining models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # defining the parameter grid
    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }

    # training rfc via GridSearchCV
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # training lrc
    lrc.fit(X_train, y_train)

    # making predictions for rfc
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # making predictions for lrc
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # generating classification report
    classification_report_image('Random Forest', y_train,
                                y_test, y_train_preds_rf, y_test_preds_rf)
    classification_report_image('Logistic Regression', y_train,
                                y_test, y_train_preds_lr, y_test_preds_lr)

    # plotting and storing roc curve
    plot_roc_curves(cv_rfc.best_estimator_, lrc, X_test, y_test, IMAGES_PATH_RESULTS)

    # plotting and storing feature importance
    feature_importance_plot(cv_rfc.best_estimator_, X, IMAGES_PATH_RESULTS)


if __name__ == "__main__":
    # performing eda
    perform_eda(df)

    # traing models
    print('The models are training...')
    train_models(X_train, X_test, y_train, y_test)