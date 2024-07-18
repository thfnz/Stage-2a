import time
import pyprind
import sys
import numpy as np
import matplotlib.pyplot as plt
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

def n_top_accuracy(alProcess, idx, idx_save, training_set):
	# Votes
	votes = []
	for idx_model in range(len(alProcess.member_sets)):
		y_pred = alProcess.member_sets[idx_model][idx]
		query = np.argsort(y_pred)[-alProcess.n_top:]
		for idx_query in query:
			votes.append([[idx_query, 1]])
	list_idx_highest_target = vote_count(votes, alProcess.n_top)

	# Descending list_idx_highest_target
	list_values_highest_target = [alProcess.y[idx] for idx in list_idx_highest_target]
	descending_list_idx_highest_target = [list_idx_highest_target[idx] for idx in np.argsort(list_values_highest_target)[::-1]]

	# n_top_accuracy
	nb_instances = len(alProcess.y)
	y_argsorted = np.argsort(alProcess.y)
	in_top = 0
	for idx_highest_target in list_idx_highest_target:
		found = False
		idx = 0
		while not found:
			if idx_highest_target == y_argsorted[idx]:
				found = True
				if idx > nb_instances - (alProcess.n_top + 1):
					in_top += 1
			else:
				idx += 1

	n_top_accuracy = (in_top / alProcess.n_top) * 100
	alProcess.class_set[idx_save].append(n_top_accuracy)
	# print('(' + training_set + ') Top ' + str(self.n_top) + ' accuracy : ' + str(n_top_accuracy) + '%')

	return descending_list_idx_highest_target

def plot_top_n_accuracy(alProcess, name = '', folder = '', display = False, save = False):
		try:
			alProcess.member_sets
		except:
			raise Exception('member_sets not initialized')
		nb_members = len(alProcess.member_sets)

		plt.figure()
		plt.plot(range(1, len(alProcess.class_set[0]) + 1), alProcess.class_set[0], label = 'X_train')
		plt.plot(range(1, len(alProcess.class_set[2]) + 1), alProcess.class_set[2], label = 'n_top')
		if len(alProcess.class_set[3]) > 0:
			plt.plot(range(1, len(alProcess.class_set[3]) + 1), alProcess.class_set[3], label = 'n_top_uncertainty')
		plt.legend()
		plt.xlabel('Iteration')
		plt.ylabel('Accuracy (%)')
		plt.title('Accuracy for the top ' + str(alProcess.n_top) + ' instances\nbatch_size : ' + str(alProcess.batch_size) + ' - batch_size_highest_value : ' + str(alProcess.batch_size_highest_value) + ' - nb_members : ' + str(nb_members))

		if display:
			plt.show()

		if save:
			check_images_dir(folder)
			path = './images/' + folder + 'plot_top_n_accuracy_' + name + '_'
			for stra in alProcess.reg_stra:
				if type(stra) == list:
					stra = stra[0]
				path += (stra + '_')
			plt.savefig(path + 'bs' + str(alProcess.batch_size) + '_bshv' + str(alProcess.batch_size_highest_value) + '_m' + str(nb_members) + '.png', dpi=300)

		plt.close()

def plot_nb_already_labeled(alProcess, name = '', folder = '', display = False, save = False):
	if len(alProcess.class_set[5]) == 0:
		raise Exception('Model has never learned')
		
	plt.figure()
	plt.plot(range(1, len(alProcess.class_set[5]) + 1), alProcess.class_set[5])
	plt.ylabel('nb_already_labeled')
	plt.xlabel('Iteration')

	if display:
		plt.show()

	if save:
		check_images_dir(folder)
		path = './images/' + folder + 'plot_nb_already_labeled' + name + '_'
		for stra in alProcess.reg_stra:
			if type(stra) == list:
				stra = stra[0]
			path += (stra + '_')
		plt.savefig(path + 'bs' + str(alProcess.batch_size) + '_bshv' + str(alProcess.batch_size_highest_value) + '_m' + str(len(alProcess.member_sets)) + '.png', dpi=300)

	plt.close()

class twoStepsNtop:

	def __init__(self, threshold, nb_iterations, batch_size = 1, batch_size_highest_value = 0, batch_size_min_uncertainty = -1, n_top = 100, n_top_train = 20):
		self.nb_iterations = nb_iterations
		self.batch_size = batch_size
		self.batch_size_highest_value = batch_size_highest_value
		self.batch_size_min_uncertainty = batch_size_min_uncertainty
		self.threshold = threshold
		self.n_top = n_top
		self.n_top_train = n_top_train
		self.class_set = [[], [], [], [], [0], []] # [[n_top_accuracy], [descending_n_top_idxs], [n_top_accuracy_n_top_train], [n_top_accuracy_n_top_train_uncertainty], [amount of self labeled instances], [nb_already_labeled]]

	def member_setsInit(self, X, y, reg_stra, nb_members, n_init, display = False):
		self.X = X
		self.y = y
		self.reg_stra = reg_stra
		self.df_size = len(y)
		self.usedToTrain = [False for i in range(self.df_size)] # Boolean representation of which data is used in X_train/y_train

		self.member_sets = []
		idx_train, idx_test = train_test_split(list(range(self.df_size)), test_size = (1 - (nb_members * n_init) / self.df_size))
		X_train, self.X_test, y_train, self.y_test = X[idx_train, :], X[idx_test, :], y[idx_train], y[idx_test]
		self.X_train_init = X_train
		for idx in idx_train:
			self.usedToTrain[idx] = True

		# Model repartition. If nb_members doesn't allow a perfect repartition, the first model of reg_stra will be used for the rest.
		for idx_reg_stra in range(len(reg_stra)):
			for idx_model in range(nb_members // len(reg_stra)):
				X_train_feature, y_train_feature, idx_init = random_training_set(X_train, y_train, n_init)
				self.member_sets.append([X_train_feature, y_train_feature, [], [], reg_stra[idx_reg_stra], [], []]) # [X_train, y_train, y_pred, r2, reg_stra, uncertainty_pred, n_top_y_pred]
				X_train, y_train = delete_data(X_train, y_train, idx_init)
		if (nb_members % len(reg_stra)) != 0:
			for idx_model in range(nb_members % len(reg_stra)):
				X_train_feature, y_train_feature, idx_init = random_training_set(X_train, y_train[:, 0], n_init)
				self.member_sets.append([X_train_feature, y_train_feature, [], [], reg_stra[0], [], []])
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
		self.class_set[1] = n_top_accuracy(self, 2, 0, 'X_train')

		# Training on the n_top_train highest predicted value
		if self.n_top_train > self.n_top:
			raise Exception('n_top_train > n_top')

		y_highest_target = self.y[self.class_set[1]]
		X_highest_target = self.X[self.class_set[1]]
		X_train, y_train = X_highest_target[-self.n_top_train:], y_highest_target[-self.n_top_train:]

		for idx_model in range(nb_members):
			model = regression_strategy(self.member_sets[idx_model][4])
			model.fit(X_train, y_train)
			self.member_sets[idx_model][6] = model.predict(self.X)

		# n_top accuracy
		n_top_accuracy(self, 6, 2, 'n_top_train')

		# Training on the n_top_train predicted value with lowest uncertainty
		# uncertainty_pred_avg
		uncertainty_pred_avg = []
		for idx in range(len(uncertainty_pred)):
			somme = 0
			for idx_model in range(nb_members):
				somme += self.member_sets[idx_model][5][idx]
			uncertainty_pred_avg.append(somme / nb_members)

		# n_top_uncertainty (new dataset containing all X_train data in the n_top + n_top_train new instances (based on uncertainty))
		idx_n_top_uncertainty_training = []

		nb_already_labeled = 0
		n_top_uncertainty = []
		for idx_n_top_relative in range(len(self.class_set[1])):
			idx_highest_target = self.class_set[1][idx_n_top_relative]
			if self.usedToTrain[idx_highest_target]:
				idx_n_top_uncertainty_training.append(idx_n_top_relative) # Keep same length (thus same idx) as all n_top sized list
				nb_already_labeled += 1
			else:
				# Relative idx (Bad practice ! :c)
				idx = 0
				found = False
				while not found and idx < len(self.X_test[:, 0]):
					if self.X[idx_highest_target, :].any() == self.X_test[idx, :].any():
						found = True
					else:
						idx += 1
				n_top_uncertainty.append([uncertainty_pred_avg[idx], idx_n_top_relative])
		self.class_set[5].append(nb_already_labeled)

		n_top_uncertainty = np.array(n_top_uncertainty)
		n_top_uncertainty_argsorted = np.argsort(n_top_uncertainty[:, 0])[::-1]
		for i in range(self.n_top_train):
			idx_n_top_uncertainty_training.append(n_top_uncertainty[n_top_uncertainty_argsorted[i], 1])
		idx_n_top_uncertainty_training = np.array(idx_n_top_uncertainty_training, dtype = int)

		# Training
		X_train, y_train = X_highest_target[idx_n_top_uncertainty_training], y_highest_target[idx_n_top_uncertainty_training]

		for idx_model in range(nb_members):
			model = regression_strategy(self.member_sets[idx_model][4])
			model.fit(X_train, y_train)
			self.member_sets[idx_model][6] = model.predict(self.X)

		# n_top accuracy
		n_top_accuracy(self, 6, 3, 'n_top_uncertainty')

		# New datasets
		# Oracle labeling
		for idx_model in range(nb_members):
			self.member_sets[idx_model][0], self.member_sets[idx_model][1] = new_datasets(self.member_sets[idx_model][0], self.member_sets[idx_model][1], self.X_test, self.y_test, final_query, [], oracle = True)
		self.X_test, self.y_test = delete_data(self.X_test, self.y_test, np.array(final_query))
		for idx in final_query:
			self.usedToTrain[idx] = True

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

		self.class_set[4].append(self.class_set[4][-1] + len(idxs_selfLabel))

		if len(idxs_selfLabel) > 0:
			for member in self.member_sets:
				member[0], member[1] = new_datasets(member[0], member[1], self.X_test, [], idxs_selfLabel, values_selfLabel, oracle = False)
			self.X_test, self.y_test = delete_data(self.X_test, self.y_test, np.array(idxs_selfLabel))
		for idx in idxs_selfLabel:
			self.usedToTrain[idx] = True

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

class twoStepsZone:

	def __init__(self, threshold, nb_iterations, batch_size_scnd_step, batch_size = 1, batch_size_highest_value = 0, batch_size_min_uncertainty = -1,n_top = 100):
		self.nb_iterations = nb_iterations
		self.batch_size_scnd_step = batch_size_scnd_step
		self.batch_size = batch_size
		self.batch_size_highest_value = batch_size_highest_value
		self.batch_size_min_uncertainty = batch_size_min_uncertainty
		self.threshold = threshold
		self.n_top = n_top
		self.class_set = [[], [], [0]] # [[n_top_accuracy], [n_top_idxs], [amount of self labeled instances]]

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
		n_top_accuracy(self, 2, 0, 'X_train')

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

		self.class_set[2].append(self.class_set[2][-1] + len(idxs_selfLabel))

		if len(idxs_selfLabel) > 0:
			for member in self.member_sets:
				member[0], member[1] = new_datasets(member[0], member[1], self.X_test, [], idxs_selfLabel, values_selfLabel, oracle = False)
			self.X_test, self.y_test = delete_data(self.X_test, self.y_test, np.array(idxs_selfLabel))

		return self.member_sets[0][0][- (self.batch_size + self.batch_size_highest_value)], self.member_sets[0][1][- (self.batch_size + self.batch_size_highest_value)]

	def scndStep(self, display = False):
		# batch_size_scnd_step plus proches voisins à label puis retrain ?
		# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
		# Appliquer sur top 1 ou X NN ?
		return

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

		scndStep(display = True)

	def initLearn(self, X, y, reg_stra, nb_members, n_init, display = False, pbar = False):
		self.member_setsInit(X, y, reg_stra, nb_members, n_init, display = display)
		self.learn(display = display, pbar = pbar)


