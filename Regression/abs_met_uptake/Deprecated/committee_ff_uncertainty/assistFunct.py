import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

def delete_data(X, y, query):
	# Avoids repetition
	X = np.delete(X, query, axis = 0)
	y = np.delete(y, query, axis = 0)
	return X, y

def random_training_set(X, y, n_init):
	# Return a random training set of n_init elements
	idx_init = np.random.choice(range(len(X)), size = n_init, replace = False)
	X_train, y_train = X[idx_init], y[idx_init]
	return X_train, y_train, idx_init

def ask_oracle(y_test, query):
	# Simulates the request to the oracle
	return y_test[query]

def new_datasets(X_train, y_train, X_test, y_test, query, ans, oracle):
	## TODO implement threshold
	# Labeling
	if oracle:
		answer = ask_oracle(y_test, query)
	else:
		answer = ans

	# New dataset
	X_train = np.concatenate((X_train, X_test[query]), axis = 0) # In this case, axis = 0 is useless because dim(X) = 1
	y_train = np.concatenate((y_train, answer))

	return X_train, y_train

def vote_count(votes, batch_size):
	# Return a final query made with all the candidates which won the election. In the event of a tie, randomly select winners.

	# Vote count for each candidate
	count = [] # = [[candidate, number of votes], ..., [candidate, number of votes]]
	for query in votes:
		for candidate in query: # candidate = [idx, weight of the vote]
			alreadyInCount = False
			i = 0
			while not alreadyInCount and i < len(count):
				if candidate[0] == count[i][0]:
					count[i][1] = count[i][1] + candidate[1]
					alreadyInCount = True
				i += 1
			if not alreadyInCount:
				count.append(candidate)

	# List all the winners
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

	return final_query

def check_images_dir(dir):
	os.makedirs('./images', exist_ok = True)
	os.makedirs(dir, exist_ok = True)

def plot_values(member_sets, X_test, y_test, X, y_pred_avg, feature_columns, n_init, batch_size, batch_size_highest_value, iteration, reg_stra, threshold, lines = 4, columns = 4, display = False, save = False):
	fig, axs = plt.subplots(lines, columns, figsize = (15, 12))
	l, c = 0, 0
	X_train, y_train = member_sets[0][0], member_sets[0][1]
	for idx_feature in range(lines * columns):
		# axs[l, c].scatter(X[:, idx_feature], y_pred, color = 'Green', label = 'Predicted data', s = 8)
		axs[l, c].scatter(X_test[:, idx_feature], y_test, color = 'Blue', label = 'Test data', alpha = 0.5, s = 1)
		#print(n_init, (batch_size + batch_size_highest_value))
		axs[l, c].scatter(X_train[n_init : len(X_train[:, 0]) - (batch_size + batch_size_highest_value - 1), idx_feature], y_train[n_init : len(X_train[:, 0]) - (batch_size + batch_size_highest_value - 1)], color = 'Black', label = 'Train data', alpha = 0.5, s = 1)
		axs[l, c].scatter(X_train[-(batch_size + batch_size_highest_value) : len(X_train[:, 0]) - batch_size_highest_value, idx_feature], y_train[-(batch_size + batch_size_highest_value) : len(X_train[:, 0]) - batch_size_highest_value], color = 'Red', label = 'Last train data added (uncertainty)', s = 4)
		# axs[l, c].scatter(X_train[-batch_size_highest_value:, idx_feature], y_train[-batch_size_highest_value:], color = 'Green', label = 'Last train data added (highest pred value)', s = 4)
		axs[l, c].scatter(X[np.argsort(y_pred_avg)[-1], idx_feature], y_pred_avg[-1], color = 'm', label = 'Highest predicted value', s = 8)
		axs[l, c].set_title(feature_columns[idx_feature])
		if l == lines - 1:
			l = 0
			c += 1
		else:
			l += 1

	if display:
		plt.show()
	if save:
		path = './images/selfLabelingInde_plot_values_'
		for stra in reg_stra:
			path += (stra + '_')
		path += 'thresh' + str(threshold) + '_bs' + str(batch_size) + '_bshv' + str(batch_size_highest_value) + '_m' + str(len(member_sets)) + '/'
		check_images_dir(path)
		plt.savefig(path + 'iteration_' + str(iteration + 1) + '.png', dpi=300)

	plt.close()

def plot_r2(member_sets, idx, batch_size, batch_size_highest_value, reg_stra, threshold, lines, columns, display = False, save = False):
	fix, axs = plt.subplots(lines, columns, figsize = (15, 12))
	l, c = 0, 0
	for idx_model in range(lines * columns):
		axs[c].plot(range(len(member_sets[idx_model][idx])), member_sets[idx_model][idx]) # !!!!!!! if lines > 1 need to be fixed, temporary change (same for plot_min_uncertainty_pred)
		axs[c].set_title('Model (' + member_sets[idx_model][5] + ') ' + str(idx_model))
		if l == lines - 1:
			l = 0
			c += 1
		else:
			l += 1

	if display:
		plt.show()
	if save:
		# check_images_dir('plot_r2/')
		path = 'images/selfLabelingInde_plot_r2_'
		for stra in reg_stra:
			path += (stra + '_')
		plt.savefig(path + 'thresh' + str(threshold) + '_bs' + str(batch_size) + '_bshv' + str(batch_size_highest_value) + '_m' + str(len(member_sets)) + '.png', dpi=300)

	plt.close()

def plot_highest_target(y_sorted, y_pred_avg, best_query, batch_size, batch_size_highest_value, iteration, nb_members, reg_stra, threshold, display = False, save = False):
	plt.figure()
	plt.scatter(range(len(y_sorted)), y_sorted, color = 'Black')
	plt.scatter(best_query, y_pred_avg[best_query], color = 'Red')
	if iteration == 0:
		plt.title('Highest target\niteration : ' + str(iteration + 1) + ' - batch_size : ' + str(batch_size) + '\nbatch_size_highest_value : ' + str(batch_size_highest_value) + ' - nb_members : ' + str(nb_members))

	if display:
		plt.show()
	if save:
		path = './images/selfLabelingInde_plot_highest_target_'
		for stra in reg_stra:
			path += (stra + '_')
		path += 'thresh' + str(threshold) + '_bs' + str(batch_size) + '_bshv' + str(batch_size_highest_value) + '_m' + str(nb_members) + '/'
		check_images_dir(path)
		plt.savefig(path + 'iteration_' + str(iteration + 1) + '.png', dpi=300)

	plt.close()

def plot_comparison_best_target(y_pred_avg, y, batch_size, batch_size_highest_value, iteration, nb_members, reg_stra, threshold, display = False, save = False):
	plt.figure()
	plt.plot(np.arange(250, 300, 1), np.arange(250, 300, 1), color = 'Red', linewidth = 2)
	plt.scatter(y_pred_avg, y, color = 'Black')
	plt.xlabel('y_pred_avg')
	plt.ylabel('y')
	if iteration == 0:
		plt.title('Comparison best targets\niteration : ' + str(iteration + 1) + ' - batch_size : ' + str(batch_size) + '\nbatch_size_highest_value : ' + str(batch_size_highest_value) + ' - nb_members : ' + str(nb_members))

	if display:
		plt.show()
		print('Mean error on the ' + str(len(y)) + ' best target values : ' + str(np.mean(np.absolute(y_pred_avg - y))))

	if save:
		path = './images/selfLabelingInde_plot_comparison_best_target_'
		for stra in reg_stra:
			path += (stra + '_')
		path += 'thresh' + str(threshold) + '_bs' + str(batch_size) + '_bshv' + str(batch_size_highest_value) + '_m' + str(nb_members) + '/'
		check_images_dir(path)
		plt.savefig(path + 'iteration_' + str(iteration + 1) + '.png', dpi=300)

	plt.close()

def plot_quality(qualities, batch_size, batch_size_highest_value, nb_members, reg_stra, threshold, display = False, save = False):
	plt.figure()
	plt.plot(range(len(qualities)), qualities)
	plt.ylim(0.90, 1.01)
	plt.xlabel('Iteration')
	plt.ylabel('Position')
	plt.title('Position of the highest predicted target value in y_argsorted\nbatch_size : ' + str(batch_size) + ' - batch_size_highest_value : ' + str(batch_size_highest_value) + ' - nb_members : ' + str(nb_members))

	if display:
		plt.show()

	if save:
		path = './images/selfLabelingInde_plot_quality_'
		for stra in reg_stra:
			path += (stra + '_')
		plt.savefig(path + 'thresh' + str(threshold) + '_bs' + str(batch_size) + '_bshv' + str(batch_size_highest_value) + '_m' + str(nb_members) + '.png', dpi=300)

	plt.close()

def plot_top_n_accuracy(accuracies, batch_size, batch_size_highest_value, nb_members, n_top, reg_stra, threshold, display = False, save = False):
	plt.figure()
	plt.plot(range(len(accuracies)), accuracies)
	plt.xlabel('Iteration')
	plt.ylabel('Accuracy')
	plt.title('Accuracy for the top ' + str(n_top) + ' instances\nbatch_size : ' + str(batch_size) + ' - batch_size_highest_value : ' + str(batch_size_highest_value) + ' - nb_members : ' + str(nb_members))

	if display:
		plt.show()

	if save:
		path = './images/selfLabelingInde_plot_top_n_accuracy_'
		for stra in reg_stra:
			path += (stra + '_')
		plt.savefig(path + 'thresh' + str(threshold) + '_bs' + str(batch_size) + '_bshv' + str(batch_size_highest_value) + '_m' + str(nb_members) + '.png', dpi=300)

	plt.close()

def plot_mean_min_uncertainty_pred(member_sets, threshold, batch_size, batch_size_highest_value, iteration, nb_members, reg_stra, display = False, save = False):
	mean_uncertainty = []

	for idx in range(len(member_sets[0][6])):
		somme = 0
		for idx_model in range(nb_members):
			somme += member_sets[idx_model][6][idx]
		mean_uncertainty.append(somme / nb_members)

	plt.figure()
	plt.scatter(range(len(mean_uncertainty)), mean_uncertainty, s = 1)
	thresh = [threshold for i in range(len(mean_uncertainty))]
	plt.plot(range(len(mean_uncertainty)), thresh, color = 'Red')
	plt.ylabel('Mean uncertainty')
	if iteration == 0:
		plt.title('Mean uncertainty\nbatch_size : ' + str(batch_size) + ' - batch_size_highest_value : ' + str(batch_size_highest_value) + ' - nb_members : ' + str(nb_members))

	if display:
		plt.show()

	if save:
		path = './images/selfLabelingInde_plot_mean_min_uncertainty_pred_'
		for stra in reg_stra:
			path += (stra + '_')
		path += 'thresh' + str(threshold) + '_bs' + str(batch_size) + '_bshv' + str(batch_size_highest_value) + '_m' + str(nb_members) + '/'
		check_images_dir(path)
		plt.savefig(path + 'iteration_' + str(iteration + 1) + '.png', dpi=300)

	plt.close()

def plot_min_uncertainty_pred(member_sets, threshold, batch_size, batch_size_highest_value, iteration, nb_members, reg_stra, lines, columns, display = False, save = False):
	fix, axs = plt.subplots(lines, columns, figsize = (15, 12))
	l, c = 0, 0
	thresh = [threshold for i in range(len(member_sets[0][6]))]
	for idx_model in range(lines * columns):
		axs[c].scatter(range(len(member_sets[idx_model][6])), member_sets[idx_model][6], s = 1)
		axs[c].plot(range(len(thresh)), thresh, color = 'Red')
		axs[c].set_title('Model (' + member_sets[idx_model][5] + ') ' + str(idx_model))
		if l == lines - 1:
			l = 0
			c += 1
		else:
			l += 1

	if display:
		plt.show()

	if save:
		path = './images/selfLabelingInde_plot_min_uncertainty_pred_'
		for stra in reg_stra:
			path += (stra + '_')
		path += 'thresh' + str(threshold) + '_bs' + str(batch_size) + '_bshv' + str(batch_size_highest_value) + '_m' + str(nb_members) + '/'
		check_images_dir(path)
		plt.savefig(path + 'iteration_' + str(iteration + 1) + '.png', dpi=300)


