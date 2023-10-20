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


class ModelOps():
    '''
    Performs model operations including:
    1) Generating a class report based on model predictions
    2) Generating a feature importance plot
    3) Saving model to a file
    '''
    def __init__(self, model=None):
        self.model = model

    def class_report(self, model_name, y_train, y_test,
                     y_train_preds, y_test_preds, PATH):
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
        # creating a folder for storing class report images
        PATH.mkdir(parents=True, exist_ok=True)

        # generating a class report
        plt.rc('figure', figsize=(10, 7))
        plt.text(0.01, 1.08, str(f'{model_name} Train'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str(f'{model_name} Test'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        class_report = PATH / f'class_report_{model_name}.png'
        plt.savefig(class_report, format='png', dpi='figure')
        plt.close()

    def feature_importance(self, X, save_path):
        '''
        Creates and stores the feature importances under the supplied path.
        Input:
            - X: pandas dataframe of X values
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
        plt.figure(figsize=(20,8))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X.shape[1]), names, rotation=90)
        feat_imp = save_path / 'Feature_Importance.png'
        plt.savefig(feat_imp, format='png', dpi='figure');

    def save_model(self, PATH):
        '''
        Saves supplied model to a file.
        Input:
            - PATH: (str) path to save a model to
        Output:
            - None
        '''
        joblib.dump(self.model, PATH)

    def roc_curve_generator(self, model1, model2, X_test, y_test, output_pth):
        '''
        Generates a ROC curve and saves an image
        to a file.
        Input:
            - model: model to plot the ROC curve for
            - X_test: test dataset inputs
            - y_test: test dataset labels
        Output:
            - None
        '''
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        model1_roc_curve = plot_roc_curve(model1, X_test, y_test, ax=ax, alpha=0.8)
        model2_roc_curve = plot_roc_curve(model2, X_test, y_test, ax=ax, alpha=0.8)
        roc_curve = output_pth / 'ROC_Curve.png'
        plt.savefig(roc_curve, format='png', dpi='figure')
        plt.close()