import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from modAL.models import ActiveLearner

X = np.random.choice(np.linspace(0, 20, 10000), size=200, replace=False).reshape(-1, 1)
y = np.sin(X) + np.random.normal(scale=0.3, size=X.shape)

# plt.figure(figsize=(10, 5))
# plt.scatter(X, y, c='k', s=20)
# plt.title('sin(x) + noise')
# plt.show()

def GP_regression_std(regressor, X):
	# Query strategy : Return instance and index of highest std-deviation 
	_, std = regressor.predict(X, return_std = True) # Returns the std-deviation & the mean
	query_idx = np.argmax(std)
	return query_idx, X[query_idx]

# Original batch of samples
n_initial = 5
initial_idx = np.random.choice(range(len(X)), size = n_initial, replace = False) # Generate n_initial samples between 0 and len(X) - 1, can't select the same index twice
X_training, y_training = X[initial_idx], y[initial_idx]

# Regressor
kernel = RBF(length_scale = 1., length_scale_bounds = (1e-2, 1e3)) + WhiteKernel(noise_level = 1, noise_level_bounds = (1e-10, 1e+1)) # WhiteKernel = noise
regressor = ActiveLearner(estimator = GaussianProcessRegressor(kernel = kernel),
	query_strategy = GP_regression_std,
	X_training = X_training.reshape(-1, 1),
	y_training = y_training.reshape(-1, 1)) # reshape(-1, 1) --> into column vector

# Plot initial regressor result
X_grid = np.linspace(0, 20, 1000)
y_pred, y_std = regressor.predict(X_grid.reshape(-1, 1), return_std = True)
y_pred, y_std = y_pred.ravel(), y_std.ravel()

# plt.figure(figsize=(10, 5))
# plt.plot(X_grid, y_pred)
# plt.fill_between(X_grid, y_pred - y_std, y_pred + y_std, alpha=0.2)
# plt.scatter(X, y, c='k', s=20)
# plt.title('Initial prediction')
# plt.show()

def active_plot(X_grid, y_pred, y_std, iteration):
	plt.figure(str(iteration + 1))
	plt.plot(X_grid, y_pred)
	oneSig = plt.fill_between(X_grid, y_pred - y_std, y_pred + y_std, alpha=0.3)
	twoSig = plt.fill_between(X_grid, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.3)
	threeSig = plt.fill_between(X_grid, y_pred - 3 * y_std, y_pred + 3 * y_std, alpha=0.3)
	plt.legend([oneSig, twoSig, threeSig], ['1σ (68%)', '2σ (95%)', '3σ (99,7%)'], loc = 'upper left')
	plt.scatter(X, y, c='k', s=20)
	plt.title('Iteration ' + str(iteration + 1))
	plt.savefig('images/iteration_' + str(iteration + 1) + '.png', dpi=300)

# Active learning process

n_queries = 20
for idx in range(n_queries):
	query_idx, query_instance = regressor.query(X)
	regressor.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))
	y_pred, y_std = regressor.predict(X_grid.reshape(-1, 1), return_std = True)
	y_pred, y_std = y_pred.ravel(), y_std.ravel()
	active_plot(X_grid, y_pred, y_std, idx)

