import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    dataset = pd.read_csv(url, header = None, encoding = 'utf-8')
except:
	print('Iris failed to load')

# select setosa and versicolor
y = dataset.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# extract sepal length and petal length
X = dataset.iloc[0:100, [0, 2]].values

# We're looking to minimise the lost function alpha(X) = (model(X) - h(X))^2 (distance ^2) (that mathly represents how right is a result given by the model)

class AdalineGD:
	# ADAptive LInear NEuron classifier.

	def __init__(self, eta = 0.1, n_iter = 50, random_state = 1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def fit(self, X, y):
		# 1.
		rgen = np.random.RandomState(self.random_state)
		self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
		self.b = np.float_(0.)
		self.losses_ = [] # Mean squared errors

		for i in range(self.n_iter):
			# 2.
			net_input = self.net_input(X)
			output = self.activation(net_input) # Identité ici (cf logReg)

			# 3.
			errors = y - output # Remove the - in the derivative of the loss function
			self.w += self.eta * 2.0 * X.T.dot(errors) / X.shape[0] # We use /X.shape to make the learning rate independant of the size of the vectors
			self.b += self.eta * 2.0 * errors

			self.losses_.append((errors ** 2).mean())

		return self


	def net_input(self, X):
		return np.dot(X, self.w) + self.b

	def activation(self, X): 
		return X

	def predict(self, X):
		return np.where(self.activation(self.net_input(X)) >= 0, 1, 0) # Condition changes nothing ???

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# ada1 = AdalineGD(n_iter=10, eta=0.1).fit(X, y)
# ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('log(Mean squared error)')
# ax[0].set_title('Adaline - Learning rate 0.1')

# ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
# ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Mean squared error')
# ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.show()

# Standardize features (Z-score Normalization) (set mean value to 0), gradient descent converges much faster w/
# https://en.wikipedia.org/wiki/Feature_scaling
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# Stochastic gradient descent for large scale datasets 

class AdalineSGD:
	def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
		self.eta = eta
		self.n_iter = n_iter
		self.w_initialized = False
		self.shuffle = shuffle
		self.random_state = random_state

	def net_input(self, X):
		return np.dot(X, self.w) + self.b

	def activation(self, X):
		return X

	def predict(self, X):
		return np.when(self.activation(self.net_input) >= 0, 1, 0)

	def shuffle(self, X, y):
		r = self.rgen.permutation(len(y)) # Generate permutation of ints from 0 to len(y)
		return X[r], Y[r] # return shuffled arrays

	def initialize_weights(self, m):
		self.rgen = np.random.RandomState(self.random_state)
		self.w = self.rgen.normal(loc = 0.0, scale = 0.01, size = m)
		self.b = np.float_(0.)
		self.w_initialized = True

	def update_weights(self, xi, target):
		# return squared error
		error = target - self.activation(self.net_input(X))
		self.w += 2 * self.eta * xi * error
		self.b += 2 * self.eta * error
		return error ** 2

	def fit(self, X, y):
		self.initialize_weights(X.shape[1])
		self.losses_ = []

		for i in range(self.n_iter):
			if self.shuffle: # shuffle avoid to form unwanted pattern because of the order
				#X, y = self.shuffle(X, y)
				print(type(X))

			losses = []

			for xi, target in zip(X, y):
				losses.append(self.update_weights(xi, target))
			self.losses_.append(np.mean(losses)) # Mean squared error

			return self

	def partial_fit(self, X, y):
		# Used for new batches without having to recompute the rest
		if not self.w_initialized:
			self.initialize_weights(X.shape[1])

		if y.ravel().shape[0] > 1:
			for xi, target in zip(X, y):
				self.update_weights(xi, target)
		else:
			self.update_weights(xi, target)

		return self

ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('figures/02_15_1.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')

plt.savefig('figures/02_15_2.png', dpi=300)
plt.show()