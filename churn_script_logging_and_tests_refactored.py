'''
A testing and logging module dedicated to test functions of
churn_library_refactored.py.

Author: Vadim Polovnikov
Date: 2023-10-06
'''

import os
import logging
import pandas as pd
import churn_library_refactored as clr

from constants import PATH
from constants import CAT_COLUMNS
from constants import KEEP_COLS


logging.basicConfig(
    filename='./logs/churn_library_refactored.log',
    level = logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_import_data(import_data, path=PATH):
    '''
    Testing import_data.

    Input:
        - import_data: (func) import_data function
    Output:
        - None
    '''
    try:
        assert isinstance(path, str)
    except AssertionError as err:
       logging.error('''Testing import_data(): wrong data type for data import path.
                     Exepcting string.''')
       raise err
    
    try:
        import_data(path)
        logging.info('Testing import_data(): successful data import.')
    except FileNotFoundError as err:
        logging.error('Testing import_data(): path is not found!')
        raise err


def test_encode_helper(encode_helper, df, cat_columns=CAT_COLUMNS):
    '''
    Testing encode_helper.

    Input:
        - df: (pd.DataFrame) dataframe
        - cat_columns: (lst) list of cat columns
    Output:
        - df: (pd.DataFrame) df with onehot-encoded features
    '''
    try:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(cat_columns, list)
    except AssertionError as err:
        logging.error('''Testing test_encode_helper(): wrong data type!
                      Make sure pd.DataFrame and list of cat features are supplied.''')
        
    try:
        encode_helper(df, cat_columns)
        logging.info('Testing test_encode_helper(): categories successfully encoded.')
    except KeyError as err:
        logging.error('''Testing test_encode_helper(): supplied list of columns
                      doesn\'t match df columns.''')
        raise err


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    Tests perform_feature_engineering() function.
    '''
    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error('''Testing perform_feature_engineering(): make sure the
                      input dataset is of pd.DataFrame format.''')
        raise err

    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        assert len(X_train.columns) and len(X_test.columns) == len(KEEP_COLS)
        logging.info('''Testing perform_feature_engineering(): feature engineering
                     successfully performed.''')
    except KeyError as err:
        logging.error('''Testing perform_feature_engineering(): make sure "Churn" column
                      is present in the input dataset.''')
        raise err
    except AssertionError as err:
        logging.error('''Testing perform_feature_engineering(): wrong number of columns!
                      Output dataset columns should be equal to KEEP_COLS.''')
        raise err


if __name__ == "__main__":
    # testing import_data
    test_import_data(clr.import_data, PATH)

    # testing data encoding
    test_encode_helper(clr.encode_helper, clr.df, CAT_COLUMNS)

    # testing perform_feature_engineering()
    test_perform_feature_engineering(clr.perform_feature_engineering, clr.df)