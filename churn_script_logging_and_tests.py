'''
Module for testing churn_library.py

Author: Vadim Polovnikov
Date: 2023-10-30
'''
import logging
import pandas as pd


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data_import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data(): SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda(): The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data(): The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, df, col):
    '''
    test perform eda function
    '''
    try:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(col, str)
    except AssertionError:
        logging.error(
            'Testing perform_eda(): incorrect data type! df has to be pandas DF and col - string.')

    try:
        perform_eda(df, col)
        logging.info(
            f'Testing perform_eda(): Histogram has been plotted ({col})')
    except KeyError:
        logging.error(
            f'Testing perform_eda(): The {col} column is missing. Failed to plot a graph.')


def test_encoder_helper(encoder_helper, category, new_cat_name, df):
    '''
    test encoder helper
    '''
    try:
        assert isinstance(category, str)
        assert isinstance(new_cat_name, str)
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error('Testing test_encoder_helper(): incorrect data type!')
        raise err

    try:
        encoder_helper(df, category, new_cat_name)
        logging.info(
            'Testing test_encoder_helper(): features succesfully transformed.')
    except KeyError as err:
        logging.error(
            f'Testing test_encoder_helper(): no such column in df ({category})!')
        raise err
    except ValueError as err:
        logging.error(
            'Testing test_encoder_helper(): Values missing in encoded column.')


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            '''Testing test_perform_feature_engineering(): Wrong data type!
            Only accepts pd.DataFrame.''')
        raise err

    try:
        perform_feature_engineering(df)
        logging.info(
            '''Testing test_perform_feature_engineering():
            Feature engineering succesfully performed.''')
    except KeyError as err:
        logging.error(
            '''Testing test_perform_feature_engineering(): Feature doesn\'t exists
            in the original dataset''')
        raise err


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    except AssertionError as err:
        logging.error(
            '''Testing test_train_models(): Incorrect data type!
            X_ have to be pd.DataFrame, y_ - pd.Series.''')
        raise err
    try:
        train_models(x_train, x_test, y_train, y_test)
        logging.info(
            'Testing train_models(): models have been succesfully trained.')
    except FileNotFoundError:
        logging.error(
            'Testing train_models(): models failed to be saved! Make sure the storage path exists.')


if __name__ == "__main__":
    # testing import_data()
    import churn_library as cl
    test_import(cl.import_data)

    # testing perform_eda()
    test_eda(cl.perform_eda, cl.df, col='Churn')
    test_eda(cl.perform_eda, cl.df, col='Customer_Age')
    test_eda(cl.perform_eda, cl.df, col='Marital_Status')
    test_eda(cl.perform_eda, cl.df, col='Total_Trans_Ct')

    # testing encoder_helper()
    for cat in cl.cat_columns:
        test_encoder_helper(cl.encoder_helper, cat, f'{cat}_Churn', cl.df)

    # testing perform_feature_engineering():
    test_perform_feature_engineering(cl.perform_feature_engineering, cl.df)

    # testing train_models()
    test_train_models(cl.train_models, cl.x_train, cl.x_test,
                      cl.y_train, cl.y_test)
