import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from mlxtend.plotting import heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Importing dataf

columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep = '\t', usecols = columns)
df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})

# Removing rows with missing values

df = df.dropna(axis = 0)

# Plot pearson product-moment correlation coefficients

# cm = np.corrcoef(df.values.T)
# hm = heatmap(cm, row_names = df.columns, column_names = df.columns)
# plt.show()

# Least squared regression model

class LinearRegressionGD:

	def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def net_input(self, X):
		return np.dot(X, self.w) + self.b

	def predict(self, X):
		return self.net_input(X)

	def fit(self, X, y):
		rgen = np.random.RandomState(self.random_state)
		self.w = rgen.normal(loc = 0., scale = 0.01, size = X.shape[1])
		self.b = np.array([0.])
		self.losses_ = []

		for i in range(self.n_iter):
			output = self.net_input(X)
			errors = (y - output)
			self.w += 2 * self.eta * X.T.dot(errors) / X.shape[0]
			self.b += 2 * self.eta * errors.mean()
			self.losses_.append((errors ** 2).mean())

		return self

# Taking and preping data

X = df[['Gr Liv Area']].values
y = df[['SalePrice']].values

sc_x = StandardScaler()
sc_y = StandardScaler() # Compared to classification prob, here we need 2 scalers

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y).flatten() # Net input cal scalar --> need to compare it with flatten array

# Results

lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# plt.plot(range(1, lr.n_iter+1), lr.losses_)
# plt.ylabel('MSE')
# plt.xlabel('Epoch')
# plt.tight_layout()
# plt.show()

def lin_regplot(X, y, model):
	plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
	plt.plot(X, model.predict(X), color='black', lw=2)    
	return

# lin_regplot(X_std, y_std, lr)
# plt.xlabel('Living area above ground (standardized)')
# plt.ylabel('Sale price (standardized)')
# plt.show()

# Predicting a Sale Price thanks to the Gr Liv Area :

# feature_std = sc_x.transform(np.array([[2500]]))
# target_std = lr.predict(feature_std)
# target_reverted = sc_y.inverse_transform(target_std.reshape(-1, 1))
# print(f'Sale price: ${target_reverted.flatten()[0]:.2f}')

# Mean absolute deviation :

def mean_absolute_deviation(data):
    return np.mean(np.abs(data - np.mean(data)))
    
# print(mean_absolute_deviation(y)) # Over 50k USD ! Not a great model at all

# Robust regression model using RANSAC 

# ransac = RANSACRegressor(LinearRegression(), max_trials = 100, min_samples = None, loss = 'absolute_error', residual_threshold = None, random_state = 123)
# ransac.fit(X, y)

# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)

# plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white',  marker='o', label='Inliers')
# plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white',  marker='s', label='Outliers')
# plt.xlabel('Living area above ground in square feet')
# plt.ylabel('Sale price in U.S. dollars')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# Evaluating the performance of lin regression models

target = 'SalePrice'
features = df.columns[df.columns != target]

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
# x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
# ax1.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
# ax2.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
# ax1.set_ylabel('Residuals')
# for ax in (ax1, ax2):
# 	ax.set_xlabel('Predicted values')
# 	ax.legend(loc='upper left')
# 	ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100, color='black', lw=2)
# plt.show()

# mse_train = mean_squared_error(y_train, y_train_pred)
# mse_test = mean_squared_error(y_test, y_test_pred)
# print(f'MSE train: {mse_train:.2f}')
# print(f'MSE test: {mse_test:.2f}')

# mae_train = mean_absolute_error(y_train, y_train_pred)
# mae_test = mean_absolute_error(y_test, y_test_pred)
# print(f'MAE train: {mae_train:.2f}')
# print(f'MAE test: {mae_test:.2f}')

# r2_train = r2_score(y_train, y_train_pred)
# r2_test =r2_score(y_test, y_test_pred)
# print(f'R^2 train: {r2_train:.2f}')
# print(f'R^2 test: {r2_test:.2f}')

# Regularized methods for regression

lasso = Lasso(alpha=1.0) # L1 (cf notes)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

# print(lasso.coef_)

# train_mse = mean_squared_error(y_train, y_train_pred)
# test_mse = mean_squared_error(y_test, y_test_pred)
# print(f'MSE train: {train_mse:.3f}, test: {test_mse:.3f}')

# train_r2 = r2_score(y_train, y_train_pred)
# test_r2 = r2_score(y_test, y_test_pred)
# print(f'R^2 train: {train_r2:.3f}, {test_r2:.3f}')

# Polynomial regression

X = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])

quadratic = PolynomialFeatures(degree = 2)
X_quad = quadratic.fit_transform(X)

lr = LinearRegression()
pr = LinearRegression() # aX^2 + bX + c

X_fit = np.arange(250, 600, 10)[:, np.newaxis]

lr.fit(X, y)
y_lin_fit = lr.predict(X_fit)

pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# plt.scatter(X, y, label='Training points')
# plt.plot(X_fit, y_lin_fit, label='Linear fit', linestyle='--')
# plt.plot(X_fit, y_quad_fit, label='Quadratic fit')
# plt.xlabel('Explanatory variable')
# plt.ylabel('Predicted or known target values')
# plt.legend(loc='upper left')
# plt.show()

# Nonlinear w/ random forests

# 1 tree

X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

# sort_idx = X.flatten().argsort()
# lin_regplot(X[sort_idx], y[sort_idx], tree)
# plt.xlabel('Living area above ground in square feet')
# plt.ylabel('Sale price in U.S. dollars')
# plt.show() 

# tree_r2 = r2_score(y, tree.predict(X))
# print(tree_r2) # = 0.51 Very bad !

# Forest

target = 'SalePrice'
features = df.columns[df.columns != target]

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

forest = RandomForestRegressor(n_estimators = 1000, criterion = 'squared_error', random_state = 1, n_jobs = -1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')

r2_train = r2_score(y_train, y_train_pred)
r2_test =r2_score(y_test, y_test_pred)
print(f'R^2 train: {r2_train:.2f}')
print(f'R^2 test: {r2_test:.2f}')

x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
ax1.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
ax2.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
ax1.set_ylabel('Residuals')
for ax in (ax1, ax2):
	ax.set_xlabel('Predicted values')
	ax.legend(loc='upper left')
	ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100, color='black', lw=2)
plt.show()

