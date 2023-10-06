'''
The module is one among others that extend churn_library.py by coverting each
function into a class.

This particular module performs data operations and combines functionnalities of
the following churn_library.py functions:
1) import_data
2) encoder_helper
3) perform_feature_engineering

Date: 2023-10-04
Author: Vadim Polovnikov
'''

# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder


class ImportData():
    '''
    Imports dataset from CSV file

    Input:
        - path: (str) dataset path
    Output:
        - dataframe: (pd.DataFrame) imported dataframe
    '''
    def __init__(self, path):
        self.path = path

    def import_from_csv(self):
        return pd.read_csv(self.path)
    

class CatEncoder():
    '''
    Takes categories and onehot-encodes them
    Input:
        - dataset: (pd.DataFrame) DataFrame
        - categories: (lst) categories list
    Output:
    - df: (pd.DataFrame): updated dataframe
    '''
    def __init__(self, df):
        self.df = df

    def onehot_encode(self, categories):
        self.categories = categories
        categories = self.df[categories]

        cat_encoder = OneHotEncoder()
        cat_encoded = cat_encoder.fit_transform(categories)
        df = pd.DataFrame()
