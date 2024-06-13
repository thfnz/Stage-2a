from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Loading iris from sklearn

iris = datasets.load_iris()
X = iris.data[:, [2, 3]] # Load petal length and width
y = iris.target

# Creating validation dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

# Printing the number of each class in y, y_train, y_test (must keep the same repartition (1/1/1) because stratified)

# print('Labels counts in y:', np.bincount(y))
# print('Labels counts in y_train:', np.bincount(y_train))
# print('Labels counts in y_test:', np.bincount(y_test))

# Z-score normalization

sc = StandardScaler()
sc.fit(X) # Goes through all the data to calculate mean and std deviation
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# print(X_train_std.mean(axis = 0), X_train_std.std(axis = 0))
# print(X_test_std.mean(axis = 0), X_test_std.std(axis = 0))

# Perceptron : 

p = Perceptron(eta0 = 0.1, random_state = 1)
p.fit(X_train_std, y_train)
y_pred = p.predict(X_test_std)
# print('Misclassified examples: %d' % (y_test != y_pred).sum(), '/', len(y_test))
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred)) # 2 ways to show acc_score
# print('Accuracy: %.3f' % p.score(X_test_std, y_test))

def plot_decision_regions(X, y, classifier, resolution=0.02, test_idx = None):
	# cf ch02_perceptron.py

	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	markers = ('o', 's', '^', 'v', '<')

	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

	y_val = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	y_val = y_val.reshape(xx1.shape)
	plt.contourf(xx1, xx2, y_val, alpha = 0.3, cmap = cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x = X[y == cl, 0],  y = X[y == cl, 1], alpha = 0.8,  c = colors[idx], marker = markers[idx],  label = f'Class {cl}',  edgecolor = 'black')

	# Highlight test examples
	if test_idx:
		X_test = X[test_idx, :]
		plt.scatter(X_test[:, 0], X_test[:, 1], c = 'None', edgecolor = 'black', alpha = 1.0, linewidth = 1, marker = 'o', s = 100, label = 'Test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# plt.figure('1')
# plot_decision_regions(X = X_combined_std, y = y_combined, classifier = p, test_idx = range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc = 'upper left')
# plt.show()

# Logistic regression

def sigmoid(z): # Mainly used in order to compute probabilites because bounded between [0, 1], differentiable but can cause a neural network to be stuck as the training stage
	return 1. / (1. + np.exp(-z))

# z = np.arange(-7, 7, 0.1)
# sigma_z = sigmoid(z)

# plt.plot(z, sigma_z)
# plt.axvline(0.0, color='k')
# plt.ylim(-0.1, 1.1)
# plt.xlabel('z')
# plt.ylabel('sigma (z)')
# plt.yticks([0.0, 0.5, 1.0])
# ax = plt.gca()
# ax.yaxis.grid(True)
# plt.show()

class LogisticRegressionGD :
	def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_state = random_state

	def net_input(self, X):
		return np.dot(X, self.w) + self.b

	def activation(self, z):
		return 1. / (1. + np.exp(- np.clip(z, -250, 250)))

	def predict(self, X):
		return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0) # !!! Condition set at 0.5 cf sigmoid graph

	def fit(self, X, y):
		rgen = np.random.RandomState(self.random_state)
		self.w = rgen.normal(loc = 0., scale = 0.01, size = X.shape[1])
		self.b = np.float_(0.)
		self.losses_ = []

		for i in range(self.n_iter):
			net_input = self.net_input(X)
			output = self.activation(net_input)
			error = y - output
			self.w += self.eta * X.T.dot(error) / X.shape[0] # cf calc in notes
			self.b += self.eta * error.mean()
			loss = (- y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))) / X.shape[0]
			self.losses_.append(loss)

		return self

X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)] # Takes only the first two classes
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

# lrgd = LogisticRegressionGD(eta = 0.3, n_iter = 1000, random_state = 1)
# lrgd.fit(X_train_01_subset, y_train_01_subset)
# plot_decision_regions(X = X_train_01_subset, y = y_train_01_subset, classifier = lrgd)
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc = 'upper left')
# plt.show()

# Under/Overfitting & Regularization

"""
for c in [-5, 0, 5]:
	lr = LogisticRegression(C = 10.**c, multi_class = 'ovr')
	lr.fit(X_train_std, y_train)
	plot_decision_regions(X = X_train_std, y = y_train, classifier = lr)
	plt.xlabel('Petal length [standardized]')
	plt.ylabel('Petal width [standardized]')
	plt.legend(loc = 'upper left')
	plt.title('C = 10 ** ' + str(c))
	plt.show()
"""

# Maximum margin class w/ SVM

svm = SVC(kernel = 'sigmoid', C = 1., random_state = 1)
svm.fit(X_train_std, y_train)

# plot_decision_regions(X_combined_std, y_combined, classifier = svm, test_idx = range(105, 150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.title('kernel = sigmoid')
# plt.show()

# Decision tree learning

# Comparing different criterion 

def entropy(p):
	return -p * np.log2(p) - (1 - p) * np.log2(1 - p) # Log base changes nothing -> we chose 2 because binary (loga(b) = logc(b)/logc(a))

def gini(p):
	return p * (1 - p) + (1 - p) * (1 - (1 - p))

def error(p):
	return 1 - np.max([p, 1 - p])

# x = np.arange(0.0, 1.0, 0.01)

# ent = [entropy(p) if p != 0 else None for p in x]
# sc_ent = [e * 0.5 if e else None for e in ent]
# err = [error(i) for i in x]

# fig = plt.figure()
# ax = plt.subplot(111)
# for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], ['Entropy', 'Entropy (scaled)', 'Gini impurity', 'Misclassification error'], ['-', '-', '--', '-.'], ['black', 'lightgray', 'red', 'green', 'cyan']):
# 	line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)

# ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
# ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
# plt.ylim([0, 1.1])
# plt.xlabel('p(i=1)')
# plt.ylabel('Impurity index')
# plt.show()

# Building the tree

tree_model = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, random_state = 1)
tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# plot_decision_regions(X_combined, y_combined, classifier = tree_model, test_idx = range(105, 150))
# plt.xlabel('Petal length [cm]')
# plt.ylabel('Petal width [cm]')
# plt.legend(loc='upper left')
# plt.show()

# feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
# tree.plot_tree(tree_model, feature_names = feature_names, filled = True)
# plt.show()

# Random forests

forest = RandomForestClassifier(n_estimators = 25, random_state = 1, n_jobs = -1) # number of trees, random_state, number of jobs //
forest.fit(X_train, y_train)

# plot_decision_regions(X_combined, y_combined, classifier = forest, test_idx = range(105, 150))
# plt.xlabel('Petal length [cm]')
# plt.ylabel('Petal width [cm]')
# plt.legend(loc='upper left')
# plt.show()

kn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
kn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier = kn, test_idx = range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
