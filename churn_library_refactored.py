'''
A class-refactored churn_library.py module.

It uses varios modules created to address concrete tasks
presented in churn_library.py.

Date: 2023-10-05
Author: Vadim Polovnikov
'''

# importing modules
from data_operations import ImportData
from data_operations import FeatureEng

# importing constans
from constants import PATH
from constants import CAT_COLUMNS
from constants import KEEP_COLS


def import_data(path):
    '''
    Imports data from path

    Input:
        - path: (str): dataset path
    Output:
        - df: (pd.DataFrame) DataFrame
    '''
    return ImportData(path).import_from_csv()


# importing data
df = import_data(PATH)

# dropping extraneous column
df.drop(labels="Unnamed: 0", axis='columns', inplace=True)

# Adding 'Churn' col based on the Attrition_Flag value
df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)


def perform_eda():
    '''
    Performs EDA and saves images.
    '''
    pass


def encode_helper(df, cat_columns=CAT_COLUMNS):
    '''
    Encodes the categorical features.
    Input:
        - df: (pd.DataFrame) dataframe
        - cat_columns: (lst) list of cat features
    Output:
        - df: (pd.DataFrame) transformed df
    '''
    # onehot-encoding cat features
    encoded_cats = FeatureEng(df).onehot_encode(cat_columns)

    # dropping cat features
    df.drop(labels=cat_columns, axis='columns', inplace=True)

    # insterting encoded columns
    df[encoded_cats.columns] = encoded_cats

    return df


def perform_feature_engineering(df):
    '''
    Encodes cat features, returns train-test split
    with selected columns in a dataset.
    Input:
        - df: (pd.DataFrame) dataframe
    Output:
        - X_train: (arr) X training data
        - X_test: (arr) X testing data
        - y_train: (arr) y training data
        - y_test: (arr) y testing data
    '''
    df = encode_helper(df, CAT_COLUMNS)
    X_train, X_test, y_train, y_test = FeatureEng(df).data_splitter(
        df,
        KEEP_COLS,
        0.3
    )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)