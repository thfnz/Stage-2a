# self = variable that points to the instance of the class
# __f__ : specific function (such as =, lenght etc..)
# _variable : private variable
# variable_ : used to avoid conflicts 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):

	def __init__(self, eta = 0.1, n_iter = 50, random_state = 1): # Constructor
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def net_input(self, X):
		# Calculate net input (wX + b) --> Model
		return np.dot(X, self.w) + self.b

	def predict(self, X):
		# Return class label in our binary case
		return np.where(self.net_input(X) >= 0.0, 1, 0) # np.where(condition, res if true, res if false)
		# Very important here : hyperplan is perendicular to the y axis ! --> net_input >= 0
		
	def fit(self, X, y):
		# Train the model
		rgen = np.random.RandomState(self.random_state) # Return class used to generate numbers, take seed
		self.w = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1]) # Random weights by distrib normal(mean (center), std deviation, size)
		self.b = np.float_(0.) # Set biais to 0 ; if unsure if variable is scalar, list or array -> use _
		self.errors_ = []

		for _ in range(self.n_iter):
			# !!! Very important here : don't forget that values are binary (1 or 0) -> descrete
			errors = 0
			for xi, target in zip(X, y): # zip joins 2
				update = self.eta * (target - self.predict(xi)) # 0 if good prediction, [1, -1] if not --> cf dessin plan en 3d
				# ax + by + cz = k --> MINUS cz = -k + ax + by !
				self.w += update * xi # Rotation
				self.b += update # Shift
				errors += int(update != 0.0) # Numbers of errors on this iteration
			self.errors_.append(errors)
        
		return self


# Reading-in the Iris data
try:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    dataset = pd.read_csv(url, header = None, encoding = 'utf-8')
except:
	print('Iris failed to load')

# Exctracting 2 classes (setosa and versicolor) (cf dataset)
y = dataset.iloc[0:100, 4].values # Purely integer-location based indexing for selection by position + delete labels
y = np.where(y == 'Iris-setosa', 0, 1) # Define setosa as class 1
X = dataset.iloc[0:100, [0, 2]].values

# Plot data
"""
plt.figure('1')
plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()
"""

# Training the perceptron model

p1 = Perceptron(eta = 0.1, n_iter = 10)
p1.fit(X, y)

# Plot results
"""
plt.figure('2')
plt.plot(range(1, len(p1.errors_) + 1), p1.errors_, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Number of updates')
plt.show()
"""

# Plot decision regions

def plot_decision_regions(X, y, classifier, resolution=0.02):
	# Color map & markers
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') # function still works on non binary cases (nb_class <= 5)
	cmap = ListedColormap(colors[:len(np.unique(y))]) # Take nb_class colors
	markers = ('o', 's', '^', 'v', '<')

	# Plot the decision surface (Only 2 features)
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # 1st feature
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 # 2nd
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)) # Create rectangular grid of our 2 given 1D array
	# meshgrid returns 2 2D array representing coordonate of each point ([X, Y])

	y_val = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) # predict(transpose(array(xx1_squished, xx2_squished)))
	y_val = y_val.reshape(xx1.shape) # Predict takes 1D array
	plt.contourf(xx1, xx2, y_val, alpha = 0.3, cmap = cmap) # Plot contour lines of each class
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	for idx, cl in enumerate(np.unique(y)): # For each index and value of each different class
		plt.scatter(x = X[y == cl, 0],  y = X[y == cl, 1], alpha = 0.8,  c = colors[idx], marker = markers[idx],  label = f'Class {cl}',  edgecolor = 'black')

plt.figure('3')
plot_decision_regions(X, y, classifier = p1)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()