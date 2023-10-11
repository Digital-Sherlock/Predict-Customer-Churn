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
from sklearn.model_selection import train_test_split


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
    

class FeatureEng():
    '''
    Takes categories and onehot-encodes them.
    
    Input:
        - dataset: (pd.DataFrame) DataFrame
        - categories: (lst) categories list
    Output:
        - df: (pd.DataFrame): updated dataframe
    '''
    def __init__(self, df):
        self.df = df
        self.__seed=42

    def onehot_encode(self, categories):
        '''
        Onehot-encodes categortical features and
        returns dataframe compirsed of these encoded
        features.

        Input:
            - categories: (lst) list of categories
        Output:
            - df_cat_encoded: (pd.DataFrame) dataframe of encoded
            categories
        '''        
        # dataframe comprised of only cat features
        print(self.df.columns)
        categories = self.df[categories]

        cat_encoder = OneHotEncoder(sparse=False)
        cat_encoded = cat_encoder.fit_transform(categories)

        # dataframe comprised of onehot-encoded cat features
        df_cat_encoded = pd.DataFrame(cat_encoded, columns=cat_encoder.get_feature_names_out(),
                          index=self.df.index)
        
        return df_cat_encoded
    

    def data_splitter(self, df, cols, test_size):
        '''
        Modifies datasets to include passed columns.\
        Splits the data for the given dataset.
        Input:
            - df: (pd.DataFrame) df to split
            - cols (lst): list of cols to include
        Output:
            - X_train: (arr) X training data
            - X_test: (arr) X testing data
            - y_train: (arr) y training data
            - y_test: (arr) y testing data
        '''
        # defining input and output variables
        X = pd.DataFrame()
        y = df['Churn']

        X[cols] = df[cols]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size,
                                                            random_state=self.__seed)
        
        return X_train, X_test, y_train, y_test