import numpy as np
import matplotlib.pyplot as plt
import pyprind
import sys

from assistFunct import *
from regressors import *

# Data
size_init = 200

# X = np.random.choice(np.linspace(0, 20, 10000), size = size_init, replace = False).reshape(-1, 1)
# y = np.sin(X) + np.random.normal(scale = 0.3, size = X.shape)

# X = np.linspace(0, 20, 1000).reshape(-1, 1)
# y = np.sin(X)/2 - ((10 - X)**2)/50 + 2 + np.random.normal(scale=0.05, size=X.shape)

X = np.linspace(0, 5, 1000).reshape(-1, 1)
y  = np.sin(5*X) + 0.5*np. sin(10*X) + 0.3 * np.cos(20*X)+ np.random.normal(scale=0.2, size=X.shape)

# plt.figure('Initial data')
# plt.scatter(X, y)
# plt.title('sin(x) + noise')
# plt.show()

# Model selection (polynom - gradBoost - randomForest)
reg_stra_test = ['polynom', 'gradBoost', 'randomForest']

for reg_stra in reg_stra_test:
	print('Testing : ' + reg_stra)

	# Random training set
	n_init = 2
	idx_init = np.random.choice(range(len(X)), size = n_init, replace = False).reshape(-1, 1)
	X_train, y_train = X[idx_init].reshape(-1, 1), y[idx_init].reshape(-1, 1)

	# Array of bool representing already used (false) or test (True) data indexes
	idx_test = np.ones(len(X), dtype = bool)
	idx_test[idx_init] = False
	X_test, y_test = X[idx_test].reshape(-1, 1), y[idx_test].reshape(-1, 1)

	# Hyperparameters
	polyDeg = 9
	alpha = 1e-3

	# Initial prediction
	y_pred, uncertainty_train, uncertainty_test = predictor(X_train, y_train, X_test, y_test, polyDeg, alpha, reg_stra, X, y, display = False)
	plot_values(X_test, y_test, X_train, y_train, X, y_pred, len(X_test), -1, reg_stra, display = False, save = True)
	plot_y_corr(y_pred, y, -1, reg_stra, display = False, save = True)

	# AL
	nb_iterations = 40
	batch_size = int(4 / 2)
	threshold = 1e-3

	# Optional : pyprind progBar
	pbar = pyprind.ProgBar(nb_iterations, stream = sys.stdout)

	for iteration in range(nb_iterations):
		# Query strategy
		uncertainty_pred = uncertainty_predictor(X_train, uncertainty_train, X_test, uncertainty_test,polyDeg, alpha, reg_stra, display = False)

		# Plot uncertainty
		plot_uncertainty(X_test, uncertainty_pred, iteration, reg_stra, display = False, save = True)

		# Extract worst and best uncertainties
		# Value
		min_uncertainty_value = np.amin(uncertainty_pred)
		# Index
		query_max_uncertainty_idx = np.argsort(uncertainty_pred)[-batch_size:] # Low confidence
		query_min_uncertainty_idx = np.argsort(uncertainty_pred)[:batch_size] # High confidence

		# Labeling
		# Low uncertainty sample
		if min_uncertainty_value < threshold and iteration > 2:
			low_uncertainy_instance = y_pred[query_min_uncertainty_idx]
		else:
			low_uncertainy_instance = y_test[query_min_uncertainty_idx]
		# High uncertainty sample
		high_uncertainty_instance = y_test[query_max_uncertainty_idx]

		# New datasets
		new_y_train = np.concatenate((low_uncertainy_instance, high_uncertainty_instance)).reshape(-1, 1)
		new_X_train = np.concatenate((X_test[query_min_uncertainty_idx], X_test[query_max_uncertainty_idx])).reshape(-1, 1)
		X_train = np.concatenate((X_train, new_X_train), axis = 0)
		y_train = np.concatenate((y_train, new_y_train))
		X_test = np.delete(X_test, np.concatenate((query_min_uncertainty_idx, query_max_uncertainty_idx)), axis = 0)
		y_test = np.delete(y_test, np.concatenate((query_min_uncertainty_idx, query_max_uncertainty_idx)))

		# Training and prediction
		y_pred, uncertainty_train, uncertainty_test = predictor(X_train, y_train, X_test, y_test, polyDeg, alpha, reg_stra, X, y, display = (iteration == nb_iterations - 1))

		# Plot prediction
		plot_values(X_test, y_test, X_train, y_train, X, y_pred, batch_size, iteration, reg_stra, display = False, save = True)
		plot_y_corr(y_pred, y, iteration, reg_stra, display = False, save = True)

		# Optional : pyprind progBar
		pbar.update()
