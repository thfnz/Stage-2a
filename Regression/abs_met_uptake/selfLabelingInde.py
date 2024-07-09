import time
import pyprind
import sys
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from regressors import regression_strategy
from assistFunct import *

### Query strategies

# Lf max value
def query_target_max_value(X_train, y_train, X_test, reg_stra, batch_size, display = False):
	# Trains the reg_stra model on a 1D-feature, tries to predict the target value on the test dataset and query what it thinks are the batch_size instances resulting in the max y
	start_time = time.time()

	# Model selection and fit
	model = regression_strategy(reg_stra)
	model.fit(X_train, y_train)
	if display:
		print('Model (' + reg_stra + ') took ' + str(time.time() - start_time) + 's to be trained on ' + str(len(y_train)) + ' intances.')

	# Prediction
	y_pred = model.predict(X_test)

	# Indice of maximum predicted value
	query = [[np.argsort(y_pred)[-i], 1]for i in range(batch_size)]

	return query

# Uncertainty
def uncertainty_sampling(X_train, y_train, X_test, y_test, X, y, threshold, reg_stra, batch_size, batch_size_min_uncertainty, display = False):
	# Predictions
	y_pred, uncertainty_train, r2_score = predictor(X_train, y_train, X_test, X, y, reg_stra, display = display)
	uncertainty_pred = uncertainty_predictor(X_train, uncertainty_train, X_test, reg_stra, display = display)
	uncertainty_pred_argsorted = np.argsort(uncertainty_pred)

	# Extract worst uncertainties
	query_max_uncertainty_idx = uncertainty_pred_argsorted[-batch_size:] # Low confidence
	query_max_uncertainty_value = uncertainty_pred[query_max_uncertainty_idx]
	query = [[query_max_uncertainty_idx[i], query_max_uncertainty_value[i]] for i in range(batch_size)]

	# Extract best uncertainties and their predicted values
	if batch_size_min_uncertainty == -1:
		batch_size_min_uncertainty = len(uncertainty_pred)

	selfLabel = []
	min_uncertainty_idx = uncertainty_pred_argsorted[:batch_size_min_uncertainty]
	for idx in min_uncertainty_idx:
		if uncertainty_pred[idx] < threshold:
			# Absolute idx
			idx_abs = 0
			found = False
			while not found and idx_abs < len(y_pred):
				if X_test[idx, :].any() == X[idx_abs, :].any():
					found = True
					# Labeling
					selfLabel.append([idx, y_pred[idx_abs]])
				else:
					idx_abs += 1

	return y_pred, query, r2_score, uncertainty_pred, selfLabel

def predictor(X_train, y_train, X_test, X, y, reg_stra, display = False):
	# Fit the chosen model and returns predicted targets (of every instances of the dataset) + the uncertainty train data
	start_time = time.time()

	# Model selection and fit
	model = regression_strategy(reg_stra)
	model.fit(X_train, y_train)
	if display:
		print('Model (' + reg_stra + ') took ' + str(time.time() - start_time) + 's to be trained with ' + str(len(y_train)) + ' intances.')

	# Prediction on the entire data set
	y_pred = model.predict(X)

	# Uncertainty on train set
	y_train_pred = model.predict(X_train)
	uncertainty_train = np.absolute(np.subtract(y_train_pred, y_train))

	# Printing values
	if display:
		print("\nMean absolute error: %.2f" % np.mean(np.absolute(y_pred - y)))
		print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y) ** 2))
		print("R2-score: %.2f" % r2_score(y, y_pred))

	return y_pred, uncertainty_train, r2_score(y, y_pred)

def uncertainty_predictor(X_train, y_train, X_test, reg_stra, display = False):
	# Returns uncertainty of the predicted targets values of X_test
	start_time = time.time()

	# Model selection and fit
	model = regression_strategy(reg_stra)
	model.fit(X_train, y_train)
	if display:
		print('Model (' + reg_stra + ') took ' + str(start_time - time.time()) + 's to be trained with ' + str(len(y_train)) + ' intances.')

	uncertainty_predicted = np.absolute(model.predict(X_test)) # y_train = uncertainty_train, very clever technique !

	return uncertainty_predicted

### AL

class selfLabelingInde:

	def __init__(self, threshold, nb_iterations, batch_size = 1, batch_size_highest_value = 0, batch_size_min_uncertainty = -1, n_top = 100):
		self.nb_iterations = nb_iterations
		self.batch_size = batch_size
		self.batch_size_highest_value = batch_size_highest_value
		self.batch_size_min_uncertainty = batch_size_min_uncertainty
		self.threshold = threshold
		self.n_top = n_top
		self.class_set = [[]] # [[n_top_accuracy]]

	def member_setsInit(self, X, y, reg_stra, nb_members, n_init, display = False):
		self.X = X
		self.y = y
		self.reg_stra = reg_stra

		self.member_sets = []
		X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size = (1 - (nb_members * n_init) / len(y)))

		# Model repartition. If nb_members doesn't allow a perfect repartition, the first model of reg_stra will be used for the rest.
		for idx_reg_stra in range(len(reg_stra)):
			for idx_model in range(nb_members // len(reg_stra)):
				X_train_feature, y_train_feature, idx_init = random_training_set(X_train, y_train, n_init)
				self.member_sets.append([X_train_feature, y_train_feature, [], [], reg_stra[idx_reg_stra], []]) # [X_train, y_train, y_pred, r2, reg_stra, uncertainty_pred]
				X_train, y_train = delete_data(X_train, y_train, idx_init)
		if (nb_members % len(reg_stra)) != 0:
			for idx_model in range(nb_members % len(reg_stra)):
				X_train_feature, y_train_feature, idx_init = random_training_set(X_train, y_train[:, 0], n_init)
				self.member_sets.append([X_train_feature, y_train_feature, [], [], reg_stra[0]], [])
				X_train, y_train = delete_data(X_train, y_train, idx_init)

		return self.member_sets, self.X_test, self.y_test

	def learnOnce(self, display = False):

		try:
			self.member_sets
		except:
			raise Exception('member_sets not initialized')
		nb_members = len(self.member_sets)

		if self.batch_size < 1 and self.batch_size_highest_value < 1:
			raise Exception('At least one batch_size must be > 1')

		# Uncertainty sampling
		if self.batch_size > 0: 
			votes = []
			selfLabels = []
			for idx_model in range(nb_members):
				X_train, y_train = self.member_sets[idx_model][0], self.member_sets[idx_model][1]

				# Vote
				y_pred, query, r2_score_y, uncertainty_pred, selfLabel = uncertainty_sampling(X_train, y_train, self.X_test, self.y_test, self.X, self.y, self.threshold, self.member_sets[idx_model][4], self.batch_size, self.batch_size_min_uncertainty, display = display)
				votes.append(query)
				selfLabels.append(selfLabel)
				self.member_sets[idx_model][2], self.member_sets[idx_model][5] = y_pred, uncertainty_pred
				self.member_sets[idx_model][3].append(r2_score_y)

			# Vote count
			final_query = vote_count(votes, self.batch_size)

		# Max value sampling
		if self.batch_size_highest_value > 0:
			votes = []
			for idx_model in range(nb_members):
				X_train, y_train = self.member_sets[idx_model][0], self.member_sets[idx_model][1]
				query = query_target_max_value(X_train, y_train, self.X_test, self.member_sets[idx_model][4], self.batch_size_highest_value, display = display)
				votes.append(query)

			for candidate in vote_count(votes, self.batch_size_highest_value):
				final_query.append(candidate)

		# n_top accuracy
		votes = []
		for idx_model in range(nb_members):
			y_pred = self.member_sets[idx_model][2]
			query = np.argsort(y_pred)[-self.n_top:]
			for idx_query in query:
				votes.append([[idx_query, 1]])
		list_idx_highest_target = vote_count(votes, self.n_top)

		nb_instances = len(self.y)
		y_argsorted = np.argsort(self.y)
		in_top = 0
		for idx_highest_target in list_idx_highest_target:
			found = False
			idx = 0
			while not found:
				if idx_highest_target == y_argsorted[idx]:
					found = True
					if idx > nb_instances - (self.n_top + 1):
						in_top += 1
				else:
					idx += 1

		n_top_accuracy = (in_top / self.n_top) * 100
		self.class_set[0].append(n_top_accuracy) # Need to be changed if plot

		if display:
			print('(selfLabelingInde) Top ' + str(self.n_top) + ' accuracy : ' + str(n_top_accuracy) + '%')

		# New datasets
		# Oracle labeling
		for idx_model in range(nb_members):
			self.member_sets[idx_model][0], self.member_sets[idx_model][1] = new_datasets(self.member_sets[idx_model][0], self.member_sets[idx_model][1], self.X_test, self.y_test, final_query, [], oracle = True)
		self.X_test, self.y_test = delete_data(self.X_test, self.y_test, np.array(final_query))

		# Self labeling
		idxs_selfLabel = []
		values_selfLabel = []
		for selfLabelModel in selfLabels: # Unpacking
			for selfLabel in selfLabelModel: # selfLabel = [idx, predicted value]
				# Avoid repetition
				alreadyQueried = False
				idx = 0
				while not alreadyQueried and idx < len(idxs_selfLabel):
					if selfLabel[0] == idxs_selfLabel[idx]:
						alreadyQueried = True
					idx += 1
				if not alreadyQueried:
					idxs_selfLabel.append(selfLabel[0])
					values_selfLabel.append(selfLabel[1])

		if len(idxs_selfLabel) > 0:
			for member in self.member_sets:
				member[0], member[1] = new_datasets(member[0], member[1], self.X_test, [], idxs_selfLabel, values_selfLabel, oracle = False)
			self.X_test, self.y_test = delete_data(self.X_test, self.y_test, np.array(idxs_selfLabel))

		return self.member_sets[0][0][- (self.batch_size + self.batch_size_highest_value)], self.member_sets[0][1][- (self.batch_size + self.batch_size_highest_value)]

	def learn(self, display = False, pbar = False):
		try:
			self.member_sets
		except:
			raise Exception('member_sets not initialized')

		if pbar:
			pbar = pyprind.ProgBar(self.nb_iterations, stream = sys.stdout)

		for iteration in range(self.nb_iterations):
			self.learnOnce(display = display)

			if pbar:
				pbar.update()

	def initLearn(self, X, y, reg_stra, nb_members, n_init, display = False, pbar = False):
		self.member_setsInit(X, y, reg_stra, nb_members, n_init, display = display)
		self.learn(display = display, pbar = pbar)


