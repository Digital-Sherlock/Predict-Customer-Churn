import os
import logging
import pandas as pd
# import churn_library_solution as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
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
		logging.error("Testing import_data(): The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda, df, col):
	'''
	test perform eda function
	'''
	try:
		assert isinstance(df, pd.DataFrame)
		assert isinstance(col, str)
	except AssertionError:
		logging.error('Testing perform_eda(): incorrect data type! df has to be pandas DF and col - string.')

	try:
		perform_eda(df, col)
		logging.info(f'Testing perform_eda(): Histogram has been plotted ({col})')
	except KeyError:
		logging.error(f'Testing perform_eda(): The {col} column is missing. Failed to plot a graph.')
		


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
		logging.info('Testing test_encoder_helper(): feaures succesfully transformed.')
	except KeyError as err:
		logging.error(f'Testing test_encoder_helper(): no such column in df ({category})!')
		#raise err
	except ValueError as err:
		logging.error('Testing test_encoder_helper(): Values missing in encoded column.')


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''
	

def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	# testing import_data()
	from churn_library import import_data
	test_import_df = test_import(import_data)	

	# testing perform_eda()
	from churn_library import perform_eda, df
	test_perform_eda_churn_col = test_eda(perform_eda, df, col='Churn')
	test_perform_eda_churn_cx_age_col = test_eda(perform_eda, df, col='Customer_Age')
	test_perform_eda_churn_marital_status_col = test_eda(perform_eda, df, col='Marital_Status')
	test_perform_eda_otal_trans_ct_col = test_eda(perform_eda, df, col='Total_Trans_Ct')

	# testing encoder_helper()
	from churn_library import encoder_helper, cat_columns
	for cat in cat_columns:
		test_encoder_helper(encoder_helper, cat, f'{cat}_Churn', df)

	# testing perform_feature_engineering():
	from churn_library import perform_feature_engineering
	test_perform_feature_engineering(perform_feature_engineering(df))







