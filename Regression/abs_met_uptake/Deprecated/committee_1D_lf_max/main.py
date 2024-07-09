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
	member_sets.append([X_train_feature, y_train_feature, 0]) # 0 = Placeholder for y_pred
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

		# Query
		y_pred, query = find_target_max_value(X_train.reshape(-1, 1), y_train, X_test[:, idx_feature].reshape(-1, 1), reg_stra, batch_size, display = False)
		votes.append(query)
		member_sets[idx_feature][2] = y_pred

	# Vote count
	count = [] # = [[candidate, number of votes], ..., [candidate, number of votes]]
	for query in votes:
		for candidate in query:
			alreadyInCount = False
			i = 0
			while not alreadyInCount and i < len(count):
				if candidate == count[i][0]:
					count[i][1] = count[i][1] + 1
					alreadyInCount = True
				i += 1
			if not alreadyInCount:
				count.append([candidate, 1])

	final_query = []
	while len(final_query) < batch_size:
		# Get a list of all the candidates with the maxVoteCount
		maxVoteCount = 0
		idx_maxVoteCount = np.array([])
		for candidate in count:
			if candidate[1] > maxVoteCount:
				maxVoteCount = candidate[1]
				idx_maxVoteCount = [candidate[0]]

			elif candidate[1] == maxVoteCount:
				idx_maxVoteCount.append(candidate[0])

		# In the event of a tie that would cause the batch_size to be exceeded, randomly select candidates.
		if (len(final_query) + len(idx_maxVoteCount)) > batch_size:
			idx_maxVoteCount = np.random.choice(idx_maxVoteCount, size = batch_size - len(final_query), replace = False)

		# Remove these candidates from the pool
		for candidate in idx_maxVoteCount:
			count.remove([candidate, maxVoteCount])

		# Add them to the final_query. 
		for candidate in idx_maxVoteCount:
			final_query.append(candidate)

	# Evaluation of the model (Mean value of y[final_query] compared to max target value)
	qualities.append(np.mean(y_test[final_query, 0]) / y_sorted[-1][0])

	"""
	# Find the indice of the best query's value in y_sorted (Bad practice ! There must be a better (and smarter) way)
	max_y = 0
	for query in final_query:
		if y_test[query][0] > max_y:
			best_query_absolute = y_test[query][1]
			max_y = y_test[query][0]
	
	found = False
	query_sorted = 0
	while not found:
		if best_query_absolute == y_sorted[query_sorted, 1]:
			found = True
		else:
			query_sorted += 1

	# Plot best query
	plot_best_query(y_sorted[:, 0], query_sorted, reg_stra, iteration, display = False, save = True)
	"""

	# New datasets
	for idx_feature in range(len(feature_columns)):
		member_sets[idx_feature][0], member_sets[idx_feature][1] = new_datasets(member_sets[idx_feature][0], member_sets[idx_feature][1], X_test[:, idx_feature], y_test[:, 0], final_query)
	X_test, y_test = delete_data(X_test, y_test, np.array(final_query))

	# Optional : pyprind progBar
	pbar.update()

# Plot evaluation (TODO : add gif)
plt.figure()
plt.plot(range(len(qualities)), qualities)
plt.show()

