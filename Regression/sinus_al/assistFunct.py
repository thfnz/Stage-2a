import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from regressors import regression_strategy

def predictor(X_train, y_train, X_test, y_test, polyDeg, alpha, reg_stra, X, y, display = False):
	# Fit the chosen model and returns predicted targets (of every instances of the dataset) + the uncertainty of test & train data
	start_time = time.time()

	# Model selection and fit
	model = regression_strategy(polyDeg, alpha, reg_stra)
	model.fit(X_train, y_train.ravel())
	if display:
		print('Model (' + reg_stra + ') took ' + str(time.time() - start_time) + 's to be trained with ' + str(len(y_train)) + ' intances.')

	# Prediction on the entire data set
	y_pred = model.predict(X)

	# Uncertainty on test set
	y_test_pred = model.predict(X_test).reshape(-1, 1)
	uncertainty_test = np.absolute(y_test_pred.ravel() - y_test.ravel())

	# Uncertainty on train set
	y_train_pred = model.predict(X_train).reshape(-1, 1)
	uncertainty_train = np.absolute(np.subtract(y_train_pred, y_train))

	# Printing values
	if display:
		print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - y)))
		print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y) ** 2))
		print("R2-score: %.2f" % r2_score(y, y_pred))

	return y_pred, uncertainty_train, uncertainty_test

def uncertainty_predictor(X_train, y_train, X_test, y_test, polyDeg, alpha, reg_stra, display = False):
	# Returns uncertainty of the predicted targets values of X_test
	start_time = time.time()

	# Model selection and fit
	model = regression_strategy(polyDeg, alpha, reg_stra)
	model.fit(X_train, y_train.ravel())
	if display:
		print('Model (' + reg_stra + ') took ' + str(start_time - time.time()) + 's to be trained with ' + str(len(y_train)) + ' intances.')

	uncertainty_predicted = np.absolute(model.predict(X_test).ravel() - y_test.ravel()) # y_train = uncertainty_train, very clever technique !

	return uncertainty_predicted

def check_images_dir(dir):
	os.makedirs('./images', exist_ok = True)
	os.makedirs('./images/' + dir, exist_ok = True)

def plot_values(X_test, y_test, X_train, y_train, X, y_pred, batch_size, iteration, reg_stra, display = False, save = False):
	check_images_dir('plot_values/' + reg_stra)

	plt.figure(str(iteration + 1))
	plt.title('Model : ' + reg_stra + ' - Iteration ' + str(iteration + 1))
	plt.scatter(X, y_pred, color = 'Red', label = 'Predicted data', s = 8)
	plt.scatter(X_test, y_test, color = 'Black', label = 'Test data', alpha = 0.5, s = 5)
	plt.scatter(X_train[-2 * batch_size:], y_train[-2 * batch_size:], color = 'Green', label = 'Last train data added', s = 8, marker = 's')
	plt.scatter(X_train[: - 2 * batch_size], y_train[: - 2 * batch_size], color = 'Blue', label = 'Train data', alpha = 0.5, s = 5)
	plt.xlabel('X')
	if iteration == -1:
		plt.legend(loc = 'upper left', prop = {'size' : 10})

	if display:
		plt.show()
	if save:
		plt.savefig('images/plot_values/' + reg_stra + '/iteration_' + str(iteration + 1) + '.png', dpi=300)

	plt.close()

def plot_y_corr(y_pred, y, iteration, reg_stra, display = False, save = False):
	check_images_dir('plot_y_corr/' + reg_stra)

	plt.figure(str(iteration + 1))
	plt.title('Model : ' + reg_stra + ' - Iteration ' + str(iteration + 1))
	plt.xlabel('y')
	plt.ylabel('y_pred')
	plt.scatter(y, y_pred)

	if display:
		plt.show()
	if save:
		plt.savefig('images/plot_y_corr/' + reg_stra + '/iteration_' + str(iteration + 1) + '.png', dpi=300)

	plt.close()

def plot_uncertainty(X_test, uncertainty, iteration, reg_stra, display = False, save = False):
	check_images_dir('plot_uncertainty/' + reg_stra)

	plt.figure(str(iteration + 1))
	plt.title('Model : ' + reg_stra + ' - Iteration ' + str(iteration + 1))
	plt.xlabel('X_test')
	plt.ylabel('Uncertainty')
	plt.scatter(X_test, uncertainty)

	if display:
		plt.show()
	if save:
		plt.savefig('images/plot_uncertainty/' + reg_stra + '/iteration_' + str(iteration + 1) + '.png', dpi=300)

	plt.close()

