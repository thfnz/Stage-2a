import numpy as np

from assistFunct import *

def uncertainty_sampling(X_train, y_train, X_test, y_test, X, y, reg_stra, batch_size, threshold, display = False):
	# Predictions
	y_pred, uncertainty_train, uncertainty_test, r2_score = predictor(X_train, y_train, X_test, y_test, reg_stra, X, y, display = display)
	uncertainty_pred = uncertainty_predictor(X_train, uncertainty_train, X_test, uncertainty_test, reg_stra, display = display)

	# Extract worst and best uncertainties
	# Value
	min_uncertainty_value = np.amin(uncertainty_pred)
	# Index
	query_max_uncertainty_idx = np.argsort(uncertainty_pred)[-batch_size:] # Low confidence
	query_min_uncertainty_idx = np.argsort(uncertainty_pred)[:batch_size] # High confidence

	# Labeling
	# Low uncertainty sample
	if min_uncertainty_value < threshold:
		low_uncertainy_instance = y_pred[query_min_uncertainty_idx]
	else:
		low_uncertainy_instance = y_test[query_min_uncertainty_idx]
	# High uncertainty sample
	high_uncertainty_instance = y_test[query_max_uncertainty_idx]

	# New datasets
	new_y_train = np.concatenate((low_uncertainy_instance, high_uncertainty_instance))
	new_X_train = np.concatenate((X_test[query_min_uncertainty_idx], X_test[query_max_uncertainty_idx]))
	X_train = np.concatenate((X_train, new_X_train), axis = 0)
	y_train = np.concatenate((y_train, new_y_train))
	X_test = np.delete(X_test, np.concatenate((query_min_uncertainty_idx, query_max_uncertainty_idx)), axis = 0)
	y_test = np.delete(y_test, np.concatenate((query_min_uncertainty_idx, query_max_uncertainty_idx)))

	return X_train, y_train, X_test, y_test, r2_score

def committee(X_train, X_test, y_test, X, y, batch_size, committee):
	# Query
	query_idx, query_instance = committee.query(X_test)

	# Labeling
	committee.teach(X = X_test[query_idx], y = y_test[query_idx])

	# New datasets
	X_train = np.delete(X_train, que)
