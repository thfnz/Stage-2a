import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyprind
import sys
from sklearn.model_selection import train_test_split

from assistFunct import *
from query_strategies import *

# X, y : full untouched dataset
# X_test, (y_test) : unlabeled dataset
# X_train, y_train : labeleded dataset

# Loading dataset
dataset = pd.read_csv('../properties.csv')

feature_columns = ['dimensions', ' supercell volume [A^3]', ' density [kg/m^3]', ' surface area [m^2/g]',
' num carbon', ' num hydrogen', ' num nitrogen', ' num oxygen', ' num sulfur', ' num silicon', 
' vertices', ' edges', ' genus', 
' largest included sphere diameter [A]', ' largest free sphere diameter [A]', ' largest included sphere along free sphere path diameter [A]']

X = dataset[feature_columns].values # 69840 instances
y = dataset[' absolute methane uptake high P [v STP/v]'].values
y_argsorted = np.argsort(y) # Used for evaluation
y = np.array([[y[i], int(i)]for i in range(len(y))]) # Add absolute indices to y

# Bad sorting of y in order to keep the absolute indices (used for evaluation)
y_sorted = np.zeros((len(y_argsorted), 2))

i = 0
for idx in y_argsorted:
	y_sorted[i] = y[idx]
	i += 1

# Model selection
reg_stra = 'randomForest'

# AL
nb_iterations = 40
batch_size = 10
# threshold = 1e-3

# Random training sets
member_sets = [] # Training datasets for each member of the committee
n_init = 5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (1 - (len(feature_columns) * n_init) / 69840))

for idx_feature in range(len(feature_columns)):
	X_train_feature, y_train_feature, idx_init = random_training_set(X_train[:, idx_feature], y_train[:, 0], n_init)
	member_sets.append([X_train_feature, y_train_feature, 0, []]) # 0, [] = Placeholder for y_pred and r2_score
	X_train, y_train = delete_data(X_train, y_train, idx_init)

# Optional : pyprind progBar
pbar = pyprind.ProgBar(nb_iterations, stream = sys.stdout)

# AL
qualities = []

for iteration in range(nb_iterations):
	votes = []
	# Calls for a vote on each model
	for idx_feature in range(len(feature_columns)):
		# Extract datasets from member_sets (note : Overwrite X_train and y_train isn't a issue because they have been already depleted from the "Random training sets" process)
		X_train, y_train = member_sets[idx_feature][0], member_sets[idx_feature][1]

		# Vote
		y_pred, query, r2_score = uncertainty_sampling(X_train.reshape(-1, 1), y_train, X_test[:, idx_feature].reshape(-1, 1), y_test[:, 0], X[:, idx_feature].reshape(-1, 1), y[:, 0], reg_stra, batch_size, display = False)
		votes.append(query)
		member_sets[idx_feature][2] = y_pred
		member_sets[idx_feature][3].append(r2_score)

	# Vote count
	final_query = vote_count(votes, batch_size)

	# Evaluation of the model (Search for the highest target value)
	votes = []
	for idx_feature in range(len(feature_columns)):
		# Extract datasets from member_sets
		y_pred = member_sets[idx_feature][2]
		query = np.argsort(y_pred)[-1]
		votes.append([query])
	idx_highest_target = vote_count(votes, 1)
	# Find the indice of the best query's value in y_sorted (Bad practice ! There must be a better (and smarter) way)
	found = False
	query_sorted = 0
	while not found:
		if idx_highest_target == y_sorted[query_sorted, 1]:
			found = True
		else:
			query_sorted += 1
	# Plot highest target
	plot_highest_target(y_sorted[:, 0], query_sorted, reg_stra, iteration, display = False, save = True)

	# Quality
	qualities.append(query_sorted / len(y))

	# New datasets
	for idx_feature in range(len(feature_columns)):
		member_sets[idx_feature][0], member_sets[idx_feature][1] = new_datasets(member_sets[idx_feature][0], member_sets[idx_feature][1], X_test[:, idx_feature], y_test[:, 0], final_query)
	X_test, y_test = delete_data(X_test, y_test, np.array(final_query))

	# Optional : pyprind progBar
	pbar.update()

# Quality 
"""
plt.figure()
plt.plot(range(len(qualities)), qualities)
plt.show()
"""

# r2
plot_r2(member_sets, feature_columns, reg_stra, display = True, save = False)
