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


logging.basicConfig(
    filename='./logs/churn_library_refactored.log',
    level = logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_import_data(import_data, path=PATH):
    '''
    Testing import_data.

    Input:
        - import_data: (func) data_import function
    Output:
        - None
    '''
    try:
        assert isinstance(path, str)
    except AssertionError as err:
       logging.error('Wrong data type for data import path. Exepcting string.')
       raise err
    
    try:
        import_data(path)
        logging.info('Successful data import.')
    except FileNotFoundError as err:
        logging.error('Path is not found!')
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
        logging.error('Wrong data type! Make sure pd.DataFrame and list of cat features are supplied.')
        
    try:
        encode_helper(df, cat_columns)
        logging.info('Categories successfully encoded.')
    except KeyError as err:
        logging.error('Supplied list of columns doesn\'t match df columns.')
        raise err




if __name__ == "__main__":
    # testing data_import
    test_import_data(clr.data_import, PATH)

    # testing data encoding
    test_encode_helper(clr.encode_helper, clr.df, CAT_COLUMNS)