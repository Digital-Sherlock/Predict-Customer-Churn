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
from data_operations import EDA
from pathlib import Path

# importing constans
from constants import PATH
from constants import CAT_COLUMNS
from constants import KEEP_COLS
from constants import IMAGES_PATH_EDA, IMAGES_PATH_RESULTS

# importing libraries
import matplotlib.pyplot as plt

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


def perform_eda(col, type, filename, PATH=IMAGES_PATH_EDA, **kwargs):
    '''
    Performs EDA and saves images.
    Input:
        - col: (str) column name
        - type: (str) plot type (library used for plotting)
        - filename: (str) image name
        - PATH: (pathlib.PosixPath) path to save an image to
    Output:
     - None
    '''
    if type == 'matplotlib':
        plot = EDA(df, PATH)
        plot.mplotter(col=col,
                      kind=kwargs['kind'],
                      xlabel=kwargs['xlabel'],
                      ylabel=kwargs['ylabel'],
                      filename=filename)
    else:
        plot = EDA(df, PATH)
        plot.splotter(kind=kwargs['kind'],
                      filename=filename,
                      col=col)


def encode_helper(dataset, cat_columns=CAT_COLUMNS):
    '''
    Encodes the categorical features.
    Input:
        - df: (pd.DataFrame) dataframe
        - cat_columns: (lst) list of cat features
    Output:
        - df: (pd.DataFrame) transformed df
    '''
    # onehot-encoding cat features
    encoded_cats = FeatureEng(dataset).onehot_encode(cat_columns)

    # dropping cat features
    dataset = dataset.drop(labels=cat_columns, axis='columns')

    # insterting encoded columns
    dataset[encoded_cats.columns] = encoded_cats

    return dataset


def perform_feature_engineering(dataset):
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
    dataset = encode_helper(dataset, CAT_COLUMNS)
    X_train, X_test, y_train, y_test = FeatureEng(dataset).data_splitter(
        dataset,
        KEEP_COLS,
        0.3
    )
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    perform_eda(col='Churn',
                type='matplotlib',
                filename='churn_hist.png',
                kind='hist',
                xlabel='Churn: 1 - yes, 0  - no',
                ylabel='Number of customers')
    perform_eda(col='Customer_Age',
                type='matplotlib',
                filename='churn_cx_age.png',
                kind='hist',
                xlabel='Customers\' age',
                ylabel='Number of Customers')
    
    # edge case
    plt.figure(figsize=(10,7))
    plt.ylabel('Number of customers: normalized')
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    churn_marital_status_pth = IMAGES_PATH_EDA / 'churn_marital_status.png'
    plt.savefig(churn_marital_status_pth, format='png', dpi='figure');

    perform_eda(col='Total_Trans_Ct',
                type='sns',
                kind='histplot',
                filename='total_trans_ct.png',)
    perform_eda(col='N/A',
                type='sns',
                kind='corr',
                filename='corr_matrix.png',)


