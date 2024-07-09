import pyprind
import sys
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from regressors import regression_strategy
from assistFunct import *

### Query strategy

def random_query(X_test, batch_size):
	return np.random.choice(range(len(X_test)), size = batch_size, replace = False)

### AL

class randomQuery:
	# Baseline

	def __init__(self, nb_iterations, batch_size = 1, n_top = 100):
		self.nb_iterations = nb_iterations
		self.batch_size = batch_size
		self.batch_size_highest_value = -1
		self.n_top = n_top
		self.class_set = [] # [[n_top_accuracy], ..., ]

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

		# Training and prediction
		for idx_model in range(nb_members):
			X_train, y_train = self.member_sets[idx_model][0], self.member_sets[idx_model][1]
			model = regression_strategy(self.member_sets[idx_model][4])
			model.fit(X_train, y_train)
			self.member_sets[idx_model][2] = model.predict(self.X)

		# Sampling 
		final_query = random_query(self.X_test, self.batch_size)

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
		self.class_set.append(n_top_accuracy) # Need to be changed if plot

		if display:
			print('(randomQuery) Top ' + str(self.n_top) + ' accuracy : ' + str(n_top_accuracy) + '%')

		# New datasets
		for idx_model in range(nb_members):
			self.member_sets[idx_model][0], self.member_sets[idx_model][1] = new_datasets(self.member_sets[idx_model][0], self.member_sets[idx_model][1], self.X_test, self.y_test, final_query, [], oracle = True)
		self.X_test, self.y_test = delete_data(self.X_test, self.y_test, np.array(final_query))

		return self.member_sets[0][0][-self.batch_size], self.member_sets[0][1][self.batch_size]		

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


