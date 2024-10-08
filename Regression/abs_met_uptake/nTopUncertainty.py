import time
import pyprind
import sys
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from regressors import regression_strategy
from assistFunct import *

### Query strategy

def n_top_uncertainty(n_top, boolRepr, X_train, y_train, X_test, y_test, X, y, reg_stra, batch_size, display = False):
	if batch_size > n_top:
		raise Exception('(learnOnce) n_top must be > to batch_size')

	# Predictions
	y_pred, uncertainty_train, r2_score = predictor(X_train, y_train, X, y, reg_stra, display = display)
	uncertainty_pred = uncertainty_predictor(X_train, uncertainty_train, X_test, reg_stra, display = display)

	# n_top_pred
	y_pred_argsorted = np.argsort(y_pred)
	idx_n_top_pred = y_pred_argsorted[-n_top:]

	# uncertainty_n_top_pred
	uncertainty_n_top_pred = np.array([[0., 0.] for i in range(n_top)])

	nb_false = 0 # cf assistFunct new_bool_repr
	for idx in range(len(boolRepr)):
		if not boolRepr[idx]:
			found = False
			idx_query = 0
			while not found and idx_query < len(idx_n_top_pred):
				if idx == idx_n_top_pred[idx_query]:
					found = True
					uncertainty_n_top_pred[idx_query, 0] = nb_false
					uncertainty_n_top_pred[idx_query, 1] = uncertainty_pred[nb_false]
				idx_query += 1

			nb_false += 1
		
	# Extract worst uncertainties
	uncertainty_n_top_pred_argsorted = np.argsort(uncertainty_n_top_pred[:, 1])[-batch_size:] # Low confidence
	query_max_uncertainty_idx = uncertainty_n_top_pred[uncertainty_n_top_pred_argsorted, 0]
	query_max_uncertainty_value = uncertainty_n_top_pred[uncertainty_n_top_pred_argsorted, 1]
	query = [[query_max_uncertainty_idx[i], query_max_uncertainty_value[i]] for i in range(batch_size)]

	return y_pred, query, r2_score, uncertainty_pred

def predictor(X_train, y_train, X, y, reg_stra, display = False):
	# Fit the chosen model and returns predicted targets (of every instances of the dataset) + the uncertainty train data
	start_time = time.time()

	# Model selection and fit
	model = regression_strategy(reg_stra)
	model.fit(X_train, y_train)
	if display:
		print('\nModel (' + reg_stra + ') took ' + str(time.time() - start_time) + 's to be trained with ' + str(len(y_train)) + ' intances.')

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
		print('\nModel (' + reg_stra + ') took ' + str(start_time - time.time()) + 's to be trained with ' + str(len(y_train)) + ' intances.')

	uncertainty_predicted = np.absolute(model.predict(X_test)) # y_train = uncertainty_train, very clever technique !

	return uncertainty_predicted

### AL

class nTopUncertainty:
	# Simple query by committee w/o any labeling made by the models.

	def __init__(self, nb_iterations, batch_size = 1, batch_size_highest_value = 0, n_top = 100):
		self.nb_iterations = nb_iterations
		self.batch_size = batch_size
		self.batch_size_highest_value = batch_size_highest_value
		self.n_top = n_top
		self.class_set = [[], [], []] # [[n_top_accuracy], [n_top_idxs], [bool representation of X_train]]

	def member_setsInit(self, X, y, reg_stra, nb_members, n_init, display = False):
		self.X = X
		self.y = y
		self.reg_stra = reg_stra

		# Boolean representation of which data is used in training (True if used, False if still in test dataset)
		n_samples = len(y)
		self.class_set[2] = [False for i in range(n_samples)]
		indices = np.arange(n_samples)

		# Train/Test split on indices
		self.member_sets = []
		idx_train, idx_test = train_test_split(indices, test_size = (1 - (nb_members * n_init) / len(y)))
		X_train, self.X_test, y_train, self.y_test = X[idx_train, :], X[idx_test, :], y[idx_train], y[idx_test]
		
		for idx in idx_train:
			self.class_set[2][idx] = True

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
		# 1 AL iteration, return the added training data

		try:
			self.member_sets
		except:
			raise Exception('(learnOnce) member_sets not initialized')
		nb_members = len(self.member_sets)

		if self.batch_size < 1 and self.batch_size_highest_value < 1:
			raise Exception('(learnOnce) At least one batch_size must be > 1')

		# Uncertainty sampling
		if self.batch_size > 0:
			votes = []
			for idx_model in range(nb_members):
				X_train, y_train = self.member_sets[idx_model][0], self.member_sets[idx_model][1]

				# Vote
				y_pred, query, r2_score_y, uncertainty_pred = n_top_uncertainty(self.n_top, self.class_set[2], X_train, y_train, self.X_test, self.y_test, self.X, self.y, self.member_sets[idx_model][4], self.batch_size, display = display)
				votes.append(query)
				self.member_sets[idx_model][2], self.member_sets[idx_model][5] = y_pred, uncertainty_pred
				self.member_sets[idx_model][3].append(r2_score_y)

			# Vote count
			final_query = np.array(vote_count(votes, self.batch_size), dtype = int)

		# n_top accuracy
		votes = []
		for idx_model in range(nb_members):
			y_pred = self.member_sets[idx_model][2]
			query = np.argsort(y_pred)[-self.n_top:]
			for idx_query in query:
				votes.append([[idx_query, 1]])
		self.class_set[1] = vote_count(votes, self.n_top)

		nb_instances = len(self.y)
		y_argsorted = np.argsort(self.y)
		in_top = 0
		for idx_highest_target in self.class_set[1]:
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
		self.class_set[0].append(n_top_accuracy)

		if display:
			print('(oracleOnly) Top ' + str(self.n_top) + ' accuracy : ' + str(n_top_accuracy) + '%')

		# New datasets
		for idx_model in range(nb_members):
			self.member_sets[idx_model][0], self.member_sets[idx_model][1] = new_datasets(self.member_sets[idx_model][0], self.member_sets[idx_model][1], self.X_test, self.y_test, final_query, [], oracle = True)
		self.X_test, self.y_test = delete_data(self.X_test, self.y_test, np.array(final_query))
		self.class_set[2] = new_bool_repr(self.class_set[2], final_query)

		return self.member_sets[0][0][- (self.batch_size + self.batch_size_highest_value)], self.member_sets[0][1][- (self.batch_size + self.batch_size_highest_value)]

	def learn(self, display = False, pbar = False):
		try:
			self.member_sets
		except:
			raise Exception('(learn) member_sets not initialized')

		if pbar:
			pbar = pyprind.ProgBar(self.nb_iterations, stream = sys.stdout)

		for iteration in range(self.nb_iterations):
			self.learnOnce(display = display)

			if pbar:
				pbar.update()

	def initLearn(self, X, y, reg_stra, nb_members, n_init, display = False, pbar = False):
		self.member_setsInit(X, y, reg_stra, nb_members, n_init, display = display)
		self.learn(display = display, pbar = pbar)


