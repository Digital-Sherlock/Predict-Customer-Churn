'''
A testing and logging module dedicated to test functions of
churn_library_refactored.py.

Author: Vadim Polovnikov
Date: 2023-10-06
'''
import logging
import pandas as pd
import churn_library_refactored as clr
from constants import path
from constants import cat_columns
from constants import keep_cols
from constants import images_path_eda
from churn_library_refactored import x_train, x_test, y_train, y_test


logging.basicConfig(
    filename='./logs/churn_library_refactored.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_import_data(import_data, input_path=path):
    '''
    Testing import_data.

    Input:
        - import_data: (func) import_data function
        - output_path: (str) dataset file path
    Output:
        - None
    '''
    try:
        assert isinstance(input_path, str)
    except AssertionError as err:
        logging.error(
            '''Testing import_data(): wrong data type for data import path.
                     Exepcting string.''')
        raise err

    try:
        import_data(path)
        logging.info('Testing import_data(): successful data import.')
    except FileNotFoundError as err:
        logging.error('Testing import_data(): path is not found!')
        raise err


def test_encode_helper(encode_helper, df, cat_columns=cat_columns):
    '''
    Testing encode_helper.

    Input:
        - encode_helper: (func) encode_helper function
        - df: (pd.DataFrame) dataframe
        - cat_columns: (lst) list of cat columns
    Output:
        - None
    '''
    try:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(cat_columns, list)
    except AssertionError as err:
        logging.error('''Testing test_encode_helper(): wrong data type!
                      Make sure pd.DataFrame and list of cat features are supplied.''')

    try:
        encode_helper(df, cat_columns)
        logging.info(
            'Testing test_encode_helper(): categories successfully encoded.')
    except KeyError as err:
        logging.error('''Testing test_encode_helper(): onehot_encode() failed!
                      Supplied list of cat columns doesn\'t match input dataset columns.''')
        raise err


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    Tests perform_feature_engineering() function.
    Input:
        - perform_feature_engineering: (func) function tested
        - df: (pd.DataFrame) df
    Output:
        - None
    '''
    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error('''Testing test_encode_helper(): wrong data type!
                      Make sure to supply pd.DataFrame.''')
        raise err

    try:
        x_train_arr, x_test_arr, y_train_arr, y_test_arr = perform_feature_engineering(
            df)
        # checking num of cols in dataset
        assert len(x_train_arr.columns) and len(
            x_test_arr.columns) == len(keep_cols)
        logging.info(
            '''Testing perform_feature_engineering(): feature engineering
                     successfully performed.''')
    except KeyError as err:
        logging.error(
            '''Testing perform_feature_engineering(): data_splitter() failed! Make sure
                      "Churn" column is present in the input dataset.''')
        raise err
    except AssertionError as err:
        logging.error(
            '''Testing perform_feature_engineering(): wrong number of columns!
                      Output dataset columns should be equal to KEEP_COLS.''')
        raise err


def test_perform_eda(perform_eda, col, type, filename,
                     PATH=images_path_eda, **kwargs):
    '''
    Tests perform_eda.
    Input:
        - perform_eda: (func) tested function
        - df: (pd.DataFrame) df
    Output:
        - None
    '''
    try:
        perform_eda(col, type, filename, PATH, **kwargs)
        logging.info(
            'Testing perform_eda(): Plots have been successfully saved.')
    except KeyError:
        logging.error(
            'Testing perform_eda(): Supplied column is missing in the dataset.')


def test_train_models(
        train_models,
        x_train_arr,
        x_test_arr,
        y_train_arr,
        y_test_arr):
    '''
    Tests train_models.
    Input:
        - train_models: (func) tested function
        - X_train, X_test, y_train, y_test: (pd.Series) training-test data
    Output:
        - None
    '''
    try:
        assert isinstance(x_train_arr, pd.DataFrame)
        assert isinstance(x_test_arr, pd.DataFrame)
        assert isinstance(y_train_arr, pd.Series)
        assert isinstance(y_test_arr, pd.Series)
    except AssertionError as err:
        logging.error('''Testing test_train_models(): Incorrect data type!
                      X_ have to be pd.DataFrame, y_ - pd.Series.''')
        raise err
    try:
        train_models(x_train_arr, x_test_arr, y_train_arr, y_test_arr)
        logging.info(
            'Testing train_models(): models have been succesfully trained.')
    except FileNotFoundError:
        logging.error(
            'Testing train_models(): models failed to be saved! Make sure the storage path exists.')


if __name__ == "__main__":
    # testing import_data
    test_import_data(clr.import_data, path)

    # testing encode_helper
    test_encode_helper(clr.encode_helper, clr.df)

    # testing perform_feature_engineering()
    test_perform_feature_engineering(clr.perform_feature_engineering, clr.df)

    # testing perform_eda
    test_perform_eda(
        clr.perform_eda,
        col='Churn',
        type='matplotlib',
        filename='churn_hist.png',
        kind='hist',
        xlabel='Churn: 1 - yes, 0  - no',
        ylabel='Number of customers')
    test_perform_eda(
        clr.perform_eda,
        col='Customer_Age',
        type='matplotlib',
        filename='churn_cx_age.png',
        kind='hist',
        xlabel='Customers\' age',
        ylabel='Number of Customers')
    test_perform_eda(
        clr.perform_eda,
        col='Total_Trans_Ct',
        type='sns',
        kind='histplot',
        filename='total_trans_ct.png',)
    test_perform_eda(
        clr.perform_eda,
        col='N/A',
        type='sns',
        kind='corr',
        filename='corr_matrix.png',)

    # testing train_models
    test_train_models(clr.train_models, x_train, x_test, y_train, y_test)
