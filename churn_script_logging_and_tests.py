import os
import logging
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
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda, df, col):
	'''
	test perform eda function
	'''
	try:
		perform_eda(df, col)
		logging.info('Histogram has been plotted')
	except KeyError:
		logging.error(f'The {col} column is missing. Failed to plot a graph.')
		


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	from churn_library import import_data
	test_import_df = test_import(import_data)	

	from churn_library import perform_eda
	from churn_library import df
	test_perform_eda_churn_col = test_eda(perform_eda, df, col='Churn')
	test_perform_eda_churn_cx_age_col = test_eda(perform_eda, df, col='CustoMer_Age')
	test_perform_eda_churn_marital_status_col = test_eda(perform_eda, df, col='Marital_Status')
	test_perform_eda_otal_trans_ct_col = test_eda(perform_eda, df, col='Total_Trans_Ct')







