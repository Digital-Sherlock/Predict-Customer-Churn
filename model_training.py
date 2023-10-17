'''
The module is one among others that extend churn_library.py by coverting each
function into a class.

This particular module performs model training and plotting the results of model
predictions. Here's what functionality from churn_library it covers:
1) classification_report_image
2) train_models
3) feature_importance_plot

Date: 2023-10-17
Author: Vadim Polovnikov
'''

import os
os.environ['QT_QPA_PLATFORM']='offscreen'

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
from pathlib import Path
import pandas as pd
import numpy as np
import shap # building and explaining graphs
import joblib # saving scikit-learn models
import matplotlib.pyplot as plt


class ModelResultsPlot():
    '''
    Plots the results of a model and saves images
    to a filesystem.
    '''
    def __init__(self, model, PATH):
        self.model = model
        self.PATH = PATH
        PATH.mkdir(parents=True, exist_ok=True)

    def class_report(self, model_name, y_train, y_test,
                     y_train_preds, y_test_preds):
        '''
        Produces classification report for training and testing results
        and stores report as image in images folder.
        Input:
            - model_name: (str) name of the model
            - y_train: training response values
            - y_test:  test response values
            - y_train_preds: training predictions
            - y_test_preds: test predictions

        Output:
             None
        '''
        plt.rc('figure', figsize=(10, 7))
        plt.text(0.01, 1.08, str(f'{model_name} Train'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str(f'{model_name} Test'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        class_report = self.PATH / f'class_report_{model_name}.png'
        plt.savefig(class_report, format='png', dpi='figure')
        plt.close()

    def feature_importance(self, X):
        '''
        Creates and stores the feature importances under the supplied path.
        Input:
            - X_data: pandas dataframe of X values
        Output:
            - None
        '''
        # Calculate feature importances
        importances = self.model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20,5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X.shape[1]), names, rotation=90);


class TrainingModel():
    pass