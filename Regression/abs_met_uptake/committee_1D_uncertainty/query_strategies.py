import time
import numpy as np
from sklearn.metrics import r2_score

from regressors import regression_strategy
from assistFunct import *

# Lf max value

def query_target_max_value(X_train, y_train, X_test, reg_stra, batch_size, display = False):
	### Useless for now, TODO : compare w/ and w/out
	# Trains the reg_stra model on a 1D-feature, tries to predict the target value on the test dataset and query what it thinks are the batch_size instances resulting in the max y
	start_time = time.time()

	# Model selection and fit
	model = regression_strategy(reg_stra)
	model.fit(X_train, y_train)
	if display:
		print('Model (' + reg_stra + ') took ' + str(time.time() - start_time) + 's to be trained on ' + str(len(y_train)) + ' intances.')

	# Prediction
	y_pred = model.predict(X_test)

	# Indice of maximum predicted value
	query = np.argsort(y_pred)[-batch_size:]

	return y_pred, query

# Uncertainty

def uncertainty_sampling(X_train, y_train, X_test, y_test, X, y, reg_stra, batch_size, display = False):
	# Predictions
	y_pred, uncertainty_train, r2_score = predictor(X_train, y_train, X_test, X, y, reg_stra, display = display)
	uncertainty_pred = uncertainty_predictor(X_train, uncertainty_train, X_test, reg_stra, display = display)

	# Extract worst uncertainties
	query_max_uncertainty_idx = np.argsort(uncertainty_pred)[-batch_size:] # Low confidence

	return y_pred, query_max_uncertainty_idx, r2_score

def predictor(X_train, y_train, X_test, X, y, reg_stra, display = False):
	# Fit the chosen model and returns predicted targets (of every instances of the dataset) + the uncertainty train data
	start_time = time.time()

	# Model selection and fit
	model = regression_strategy(reg_stra)
	model.fit(X_train, y_train)
	if display:
		print('Model (' + reg_stra + ') took ' + str(time.time() - start_time) + 's to be trained with ' + str(len(y_train)) + ' intances.')

	# Prediction on the entire data set
	y_pred = model.predict(X)

	# Uncertainty on train set
	y_train_pred = model.predict(X_train)
	uncertainty_train = np.absolute(np.subtract(y_train_pred, y_train))

	# Printing values
	if display:
		print("\nMean absolute error: %.2f" % np.mean(np.absolute(y_pred - y)))
		print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y) ** 2))
		print("R2-score: %.2f" % r2_score(y, y_pred))

	return y_pred, uncertainty_train, r2_score(y, y_pred)

def uncertainty_predictor(X_train, y_train, X_test, reg_stra, display = False):
	# Returns uncertainty of the predicted targets values of X_test
	start_time = time.time()

	# Model selection and fit
	model = regression_strategy(reg_stra)
	model.fit(X_train, y_train)
	if display:
		print('Model (' + reg_stra + ') took ' + str(start_time - time.time()) + 's to be trained with ' + str(len(y_train)) + ' intances.')

	uncertainty_predicted = np.absolute(model.predict(X_test)) # y_train = uncertainty_train, very clever technique !

	return uncertainty_predicted

