'''
A class-refactored churn_library.py module.

It uses varios modules created to address concrete tasks
presented in churn_library.py.

Date: 2023-10-05
Author: Vadim Polovnikov
'''

# importing modules
from data_operations import ImportData
from data_operations import CatEncoder

# importing constans
from constants import PATH
from constants import CAT_COLUMNS


def data_import(path):
    '''
    Imports data from path.

    Input:
        - path: (str): dataset path

    Output:
        - df: (pd.DataFrame) DataFrame
    '''
    return ImportData(path).import_from_csv()


# importing data
df = data_import(PATH)

# dropping extraneous column
df.drop(labels="Unnamed: 0", axis='columns', inplace=True)

# Adding 'Churn' col based on the Attrition_Flag value
df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)





if __name__ == '__main__':
    encoded_features = CatEncoder(df).onehot_encode(CAT_COLUMNS)
    print(encoded_features)