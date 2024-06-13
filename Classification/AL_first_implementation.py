from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
dataset = pd.read_csv(url, header = None, encoding = 'utf-8')

# Mapping
dataset.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
mapping = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
dataset['class'] = dataset['class'].map(mapping)

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size = 0.8, stratify = y)
X_unlabeled, X_test, y_unlabeled, y_test = train_test_split(X_unlabeled, y_unlabeled, test_size = 0.2, stratify = y_unlabeled)

stdsc = StandardScaler()
X_unlabeled = stdsc.fit_transform(X_unlabeled)
X_labeled = stdsc.fit_transform(X_labeled)
X_test = stdsc.fit_transform(X_test)

lr = LogisticRegression()
lr.fit(X_labeled, y_labeled)

# Active learning loop : 

num_iterations = 10
batch_size = 10

def plot_decision_regions(X, y, classifier, resolution=0.02):
	mapping_bckwrd = inv_size_mapping = {v: k for k, v in mapping.items()}

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
		plt.scatter(x = X[y == cl, 0],  y = X[y == cl, 1], alpha = 0.8,  c = colors[idx], marker = markers[idx],  label = mapping_bckwrd[cl],  edgecolor = 'black')

acc_score = []

for iteration in range(num_iterations):
	# Uncertainty sampling 
	uncertainty = lr.predict_proba(X_unlabeled)
	uncertainty_scores = np.max(uncertainty, axis = 1) # Max of each line
	query_indices = np.argsort(uncertainty_scores)[-batch_size:] # first batch_size elements of the array

	# Label the selected instances
	labeled_instances = X_unlabeled[query_indices]
	labeled_labels = y_unlabeled[query_indices]

	# Update the labeled and unlabeled datasets
	X_labeled = np.concatenate((X_labeled, labeled_instances), axis = 0)
	y_labeled = np.concatenate((y_labeled, labeled_labels), axis = 0)
	X_unlabeled = np.delete(X_unlabeled, query_indices, axis = 0)
	y_unlabeled = np.delete(y_unlabeled, query_indices, axis = 0)

	# Retrain the model
	lr.fit(X_labeled, y_labeled)

	# Evaluate
	validation_accuracy = accuracy_score(y_test, lr.predict(X_test))
	acc_score.append(validation_accuracy)
	print(f"Iteration {iteration+1}, Validation Accuracy: {validation_accuracy:.4f}")

	# Print
	plt.figure(str(iteration))
	plot_decision_regions(X_labeled, y_labeled, classifier = lr)
	plt.xlabel('Petal length [standardized]')
	plt.ylabel('Petal width [standardized]')
	plt.title('Iteration : ' + str(iteration + 1) + ' - Accuracy Score : ' + str(validation_accuracy))
	plt.legend(loc='upper left')
	plt.tight_layout()
	plt.savefig('images/iteration_' + str(iteration + 1) + '.png', dpi=300)


plt.figure('Acc_score')
plt.xlabel('Iteration')
plt.ylabel('Accuracy score')
plt.plot(np.arange(1, 11), acc_score)
plt.show()

