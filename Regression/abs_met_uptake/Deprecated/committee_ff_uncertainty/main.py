import pandas as pd
import numpy as np
import pyprind
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from assistFunct import *
from query_strategies import *

warnings.filterwarnings("ignore") # Skip convergence warnings

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

# Model selection (randomForest - elasticNet - elasticNetCV - XGB - SVR)
reg_stra = ['XGB']

# AL (randomForest : iter = 10, batch_size = 10, n_init = 50 - elasticNet)
nb_iterations = 130
batch_size = 1
batch_size_highest_value = 0
batch_size_min_uncertainty = -1
threshold = 1e-3

# Random training sets
nb_members = 2
member_sets = [] # Training datasets for each member of the committee
n_init = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (1 - (nb_members * n_init) / 69840)) # TODO : remove flat number

for idx_reg_stra in range(len(reg_stra)): # Model repartition. If nb_members doesn't allow a perfect repartition, the first model of reg_stra will be used for the rest.
	for idx_model in range(nb_members // len(reg_stra)):
		X_train_feature, y_train_feature, idx_init = random_training_set(X_train, y_train[:, 0], n_init)
		member_sets.append([X_train_feature, y_train_feature, 0, [], [], reg_stra[idx_reg_stra], []]) # 0, [] = Placeholder for y_pred and r2_scores
		X_train, y_train = delete_data(X_train, y_train, idx_init)
if (nb_members % len(reg_stra)) != 0:
	for idx_model in range(nb_members % len(reg_stra)):
		X_train_feature, y_train_feature, idx_init = random_training_set(X_train, y_train[:, 0], n_init)
		member_sets.append([X_train_feature, y_train_feature, 0, [], [], reg_stra[0], []]) # 0, [] = Placeholder for y_pred and r2_scores
		X_train, y_train = delete_data(X_train, y_train, idx_init)

# Optional : pyprind progBar
pbar = pyprind.ProgBar(nb_iterations, stream = sys.stdout)

# AL
accuracies = []
qualities = []
member_uncertainty_pred_n_m_one = [0 for i in range(nb_members)]

for iteration in range(nb_iterations):
	votes = []
	selfLabels = []
	# Calls for a vote on each model
	for idx_model in range(nb_members):
		# Extract datasets from member_sets (note : Overwrite X_train and y_train isn't a issue because they have been already depleted from the "Random training sets" process)
		X_train, y_train = member_sets[idx_model][0], member_sets[idx_model][1]

		# Vote
		y_pred, query, r2_score_y, uncertainty_pred, selfLabel = uncertainty_sampling(X_train, y_train, X_test, y_test[:, 0], X, y[:, 0], threshold, member_sets[idx_model][5], 1, batch_size, batch_size_min_uncertainty, display = False)
		votes.append(query)
		member_sets[idx_model][2], member_sets[idx_model][6] = y_pred, uncertainty_pred
		member_sets[idx_model][3].append(r2_score_y)

		selfLabels.append(selfLabel)

		# r2_score on train only
		# if iteration > 0:
		# 	member_sets[idx_feature][4].append(r2_score(uncertainty_pred.reshape(-1, 1), np.delete(member_uncertainty_pred_n_m_one[idx_feature], final_query).reshape(-1, 1)))
		# member_uncertainty_pred_n_m_one[idx_feature] = uncertainty_pred

	# Vote count
	final_query = vote_count(votes, batch_size)

	# Query highest predicted value
	votes = []
	for idx_model in range(nb_members):
		X_train, y_train = member_sets[idx_model][0], member_sets[idx_model][1]
		query = query_target_max_value(X_train, y_train, X_test, member_sets[idx_model][5], 1, batch_size_highest_value, display = False)
		votes.append(query)
	for candidate in vote_count(votes, batch_size_highest_value):
		final_query.append(candidate)

	# Plot y_true(y_pred_avg) for the n_top best target values
	n_top = 100
	y_pred_avg = []
	for i in range(len(y_pred)):
		somme = 0
		for idx_model in range(nb_members):
			somme += member_sets[idx_model][2][i]
		y_pred_avg.append(somme / nb_members)
	plot_comparison_best_target(np.array(y_pred_avg)[np.argsort(y_pred_avg)[-n_top:]], y_sorted[-n_top:, 0], batch_size, batch_size_highest_value, iteration, nb_members, reg_stra, threshold, display = False, save = True)

	# Evaluation of the model (Search for the highest target value)
	votes = []
	for idx_model in range(nb_members):
		# Extract datasets from member_sets
		y_pred = member_sets[idx_model][2]
		query = np.argsort(y_pred)[-n_top:]
		for idx_query in query:
			votes.append([[idx_query, 1]])
	list_idx_highest_target = vote_count(votes, n_top)
	# Find the indice of the best query's value in y_sorted (Bad practice ! There must be a better (and smarter) way
	in_top = 0
	for idx_highest_target in list_idx_highest_target:
		found = False
		query_sorted = 0
		while not found:
			if idx_highest_target == y_sorted[query_sorted, 1]:
				found = True
				if query_sorted > len(y[:, 0]) - (n_top + 1):
					in_top += 1
			else:
				query_sorted += 1
	accuracies.append(in_top)
	print('Top ' + str(n_top) + ' accuracy (iteration ' + str(iteration + 1) + ') : ' + str((in_top / n_top) * 100) + '%')

	# Plot highest target
	plot_highest_target(y_sorted[:, 0], y_pred_avg, query_sorted, batch_size, batch_size_highest_value, iteration, nb_members, reg_stra, threshold, display = False, save = True)

	# Quality
	qualities.append(query_sorted / len(y))

	# New datasets
	# Oracle labeling
	for idx_model in range(nb_members):
		member_sets[idx_model][0], member_sets[idx_model][1] = new_datasets(member_sets[idx_model][0], member_sets[idx_model][1], X_test, y_test[:, 0], final_query, [], oracle = True)
	X_test, y_test = delete_data(X_test, y_test, np.array(final_query))

	idxs_selfLabel = []
	values_selfLabel = []

	# Self labeling
	for selfLabelModel in selfLabels: # Unpacking (once again, bad practice !), worst code ever written
		for selfLabel in selfLabelModel: # selfLabel = [idx, value]
			# Avoid repetition
			alreadyQueried = False
			cpt = 0
			while not alreadyQueried and cpt < len(idxs_selfLabel):
				if selfLabel[0] == idxs_selfLabel[cpt]:
					alreadyQueried = True
				cpt += 1
			if not alreadyQueried:
				idxs_selfLabel.append(selfLabel[0])
				values_selfLabel.append(selfLabel[1])

	if len(idxs_selfLabel) > 0:
		for member in member_sets:
			member[0], member[1] = new_datasets(member[0], member[1], X_test, [], idxs_selfLabel, values_selfLabel, oracle = False)
		X_test, y_test = delete_data(X_test, y_test, np.array(idxs_selfLabel))

	# Plot values
	plot_values(member_sets, X_test, y_test[:, 0], X, y_pred_avg, feature_columns, n_init, batch_size, batch_size_highest_value, iteration, reg_stra, threshold, lines = 4, columns = 4, display = False, save = True)

	# Plot min uncertainty
	plot_min_uncertainty_pred(member_sets, threshold, batch_size, batch_size_highest_value, iteration, nb_members, reg_stra, lines = 1, columns = 2, display = False, save = True)
	plot_mean_min_uncertainty_pred(member_sets, threshold, batch_size, batch_size_highest_value, iteration, nb_members, reg_stra, display = False, save = False)

	# Optional : pyprind progBar
	# pbar.update()

# Accuracies
plot_top_n_accuracy(accuracies, batch_size, batch_size_highest_value, nb_members, n_top, reg_stra, threshold, display = False, save = True)

# Quality 
plot_quality(qualities, batch_size, batch_size_highest_value, nb_members, reg_stra, threshold, display = False, save = False)

# r2
plot_r2(member_sets, 3, batch_size, batch_size_highest_value, reg_stra, threshold, lines = 1, columns = 2, display = False, save = True)
# plot_r2(member_sets, 4, display = False, save = False)


