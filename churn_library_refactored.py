'''
A class-refactored churn_library.py module.

It uses varios modules created to address concrete tasks
presented in churn_library.py.

Date: 2023-10-05
Author: Vadim Polovnikov
'''
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from data_operations import ImportData
from data_operations import FeatureEng
from data_operations import EDA
from model_training import ModelOps
from constants import path
from constants import cat_columns
from constants import keep_cols
from constants import images_path_eda, images_path_results


def import_data(filepath):
    '''
    Imports data from path

    Input:
        - filepath: (str): dataset path
    Output:
        - df: (pd.DataFrame) DataFrame
    '''
    return ImportData(filepath).import_from_csv()


# importing data
df = import_data(path)

# dropping extraneous column
df.drop(labels="Unnamed: 0", axis='columns', inplace=True)

# Adding 'Churn' col based on the Attrition_Flag value
df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)


def perform_eda(col, type, filename, output_path=images_path_eda, **kwargs):
    '''
    Performs EDA and saves images.
    Input:
        - col: (str) column name
        - type: (str) plot type (library used for plotting)
        - filename: (str) image name
        - output_path: (pathlib.PosixPath) path to save an image to
    Output:
     - None
    '''
    # plotting matplotlib plot
    if type == 'matplotlib':
        plot = EDA(df, output_path)
        plot.mplotter(col=col,
                      kind=kwargs['kind'],
                      xlabel=kwargs['xlabel'],
                      ylabel=kwargs['ylabel'],
                      filename=filename)
    else:
    # plotting seaborn plot
        plot = EDA(df, output_path)
        plot.splotter(kind=kwargs['kind'],
                      filename=filename,
                      col=col)


def encode_helper(dataset, cat_columns=cat_columns):
    '''
    Encodes the categorical features.
    Input:
        - dataset: (pd.DataFrame) dataframe
        - cat_columns: (lst) list of cat features
    Output:
        - dataset: (pd.DataFrame) transformed df
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
        - dataset: (pd.DataFrame) dataframe
    Output:
        - x_train_arr: (arr) X training data
        - x_test_arr: (arr) X testing data
        - y_train_arr: (arr) y training data
        - y_test_arr: (arr) y testing data
    '''
    # one-hot encoding data
    dataset = encode_helper(dataset, cat_columns)
    # splitting the data
    x_train_arr, x_test_arr, y_train_arr, y_test_arr = FeatureEng(dataset).data_splitter(
        dataset,
        keep_cols,
        0.3
    )
    return x_train_arr, x_test_arr, y_train_arr, y_test_arr


x_train, x_test, y_train, y_test = perform_feature_engineering(df)


def classification_report_image(model, model_name, y_train_arr, y_test_arr,
                                y_train_preds, y_test_preds,
                                path=images_path_results):
    '''
    Produces classification report for training and testing results
    and stores report as image in images folder.
    Input:
        - model_name: (str) name of the model
        - y_train_arr: training response values
        - y_test_arr:  test response values
        - y_train_preds: training predictions
        - y_test_preds: test predictions
        - path: (pathlib.PosixPath) path to save an image to
    Output:
            None
    '''
    ModelOps(model).class_report(model_name,
                                 y_train_arr,
                                 y_test_arr,
                                 y_train_preds,
                                 y_test_preds,
                                 path)


def plot_roc_curves(model1, model2, x_test_arr, y_test_arr,
                    output_pth):
    '''
    Generates a ROC curve and saves an image
    to a file.
    Input:
        - model1, model2: model to plot the ROC curve for
        - x_test_arr: test dataset inputs
        - y_test_arr: test dataset labels
        - output_pth: path to save the plots
    Output:
        - None
    '''
    ModelOps().roc_curve_generator(model1,
                                   model2,
                                   x_test_arr,
                                   y_test_arr,
                                   output_pth)


def feature_importance(model, x_data, save_path):
    '''
    Creates and stores the feature importances under the supplied path.
    Input:
        - x_data: (pd.Dataframe) pandas dataframe of X values
        - model: model to use for plotting
        - save_path: (str) output path
    Output:
        - None
    '''
    ModelOps(model).feature_importance(x_data, save_path)


def train_models(x_train_arr, x_test_arr, y_train_arr, y_test_arr):
    '''
    The function trains models and saves the best ones.
    input:
              x_train_arr: X training data
              x_test_arr: X testing data
              y_train_arr: y training data
              y_test_arr: y testing data
    output:
              None
    '''
    # defining models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # defining the parameter grid
    param_grid = {
    'n_estimators': [200, 500],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }

    # training rfc via GridSearchCV
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train_arr, y_train_arr)

    # training lrc
    lrc.fit(x_train_arr, y_train_arr)

    # making predictions for rfc
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train_arr)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test_arr)

    # making predictions for lrc
    y_train_preds_lr = lrc.predict(x_train_arr)
    y_test_preds_lr = lrc.predict(x_test_arr)

    # saving the best performing model(s)
    ModelOps(cv_rfc.best_estimator_).save_model('./models/rfc_model.pkl')
    ModelOps(lrc).save_model('./models/logistic_model.pkl')

    # creating class report for both models
    classification_report_image(cv_rfc.best_estimator_,
                                'Random Forest',
                                y_train_arr,
                                y_test_arr,
                                y_train_preds_rf,
                                y_test_preds_rf,
                                path=images_path_results)
    classification_report_image(lrc, 'Logistic Regression',
                                y_train_arr,
                                y_test_arr,
                                y_train_preds_lr,
                                y_test_preds_lr,
                                path=images_path_results)

    # generating a ROC curve
    plot_roc_curves(
        model1=cv_rfc.best_estimator_,
        model2=lrc,
        x_test_arr=x_test_arr,
        y_test_arr=y_test_arr,
        output_pth=images_path_results)

    # generating feature importance'
    feature_importance(cv_rfc.best_estimator_, x_train_arr, images_path_results)


if __name__ == '__main__':
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
    churn_marital_status_pth = images_path_eda / 'churn_marital_status.png'
    plt.savefig(churn_marital_status_pth, format='png', dpi='figure')
    plt.close()

    perform_eda(col='Total_Trans_Ct',
                type='sns',
                kind='histplot',
                filename='total_trans_ct.png')
    perform_eda(col='N/A',
                type='sns',
                kind='corr',
                filename='corr_matrix.png')

    # training models
    train_models(x_train, x_test, y_train, y_test)
