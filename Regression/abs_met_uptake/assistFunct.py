import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from regressors import regression_strategy

def predictor(X_train, y_train, X_test, y_test, reg_stra, X, y, display = False):
	# Fit the chosen model and return predicted targets (of every instances of the dataset) + the uncertainty of test & train data
	start_time = time.time()

	# Model selection and fit
	model = regression_strategy(reg_stra)
	model.fit(X_train, y_train)
	if display:
		print('Model (' + reg_stra + ') took ' + str(time.time() - start_time) + 's to be trained with ' + str(len(y_train)) + ' intances.')

	# Prediction on the entire data set
	y_pred = model.predict(X)

	# Uncertainty on test set
	y_test_pred = model.predict(X_test)
	uncertainty_test = np.absolute(y_test_pred - y_test)

	# Uncertainty on train set
	y_train_pred = model.predict(X_train)
	uncertainty_train = np.absolute(np.subtract(y_train_pred, y_train))

	# Printing values
	if display:
		print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - y)))
		print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y) ** 2))
		print("R2-score: %.2f" % r2_score(y, y_pred))

	return y_pred, uncertainty_train, uncertainty_test, r2_score(y, y_pred)

def uncertainty_predictor(X_train, y_train, X_test, y_test, reg_stra, display = False):
	# Returns uncertainty of the predicted targets values of X_test
	start_time = time.time()

	# Model selection and fit
	model = regression_strategy(reg_stra)
	model.fit(X_train, y_train)
	if display:
		print('Model (' + reg_stra + ') took ' + str(start_time - time.time()) + 's to be trained with ' + str(len(y_train)) + ' intances.')

	uncertainty_predicted = np.absolute(model.predict(X_test) - y_test)

	return uncertainty_predicted