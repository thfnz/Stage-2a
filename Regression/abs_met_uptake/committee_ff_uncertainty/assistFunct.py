import numpy as np
import matplotlib.pyplot as plt
import os

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

def new_datasets(X_train, y_train, X_test, y_test, query):
	## TODO implement threshold
	# Labeling
	answer = ask_oracle(y_test, query)

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
	os.makedirs('./images/' + dir, exist_ok = True)

def plot_values(member_sets, X_test, y_test, X, feature_columns, reg_stra, batch_size, iteration, lines = 4, columns = 4, display = False, save = False):
	check_images_dir('plot_values/' + reg_stra)

	fig, axs = plt.subplots(lines, columns)
	l, c = 0, 0
	for idx_feature in range(len(feature_columns)):
		X_train, y_train, y_pred = member_sets[idx_feature][0], member_sets[idx_feature][1], member_sets[idx_feature][2]
		axs[l, c].scatter(X[:, idx_feature], y_pred, color = 'Red', label = 'Predicted data', s = 8)
		axs[l, c].scatter(X_test[:, idx_feature], y_test, color = 'Black', label = 'Test data', alpha = 0.5, s = 5)
		axs[l, c].scatter(X_train[: - 2 * batch_size], y_train[: - 2 * batch_size], color = 'Blue', label = 'Train data', alpha = 0.5, s = 5)
		axs[l, c].scatter(X_train[-2 * batch_size:], y_train[-2 * batch_size:], color = 'Green', label = 'Last train data added', s = 10)
		axs[l, c].set_title(feature_columns[idx_feature])
		if l == lines - 1:
			l = 0
			c += 1
		else:
			l += 1

	if display:
		plt.show()
	if save:
		plt.savefig('images/plot_values/' + reg_stra + '/iteration_' + str(iteration + 1) + '.png', dpi=300)

	plt.close()

def plot_r2(r2_scores, idx, feature_columns, reg_stra, lines = 4, columns = 4, display = False, save = True):
	check_images_dir('plot_r2/' + reg_stra)

	fix, axs = plt.subplots(lines, columns)
	l, c = 0, 0
	for idx_feature in range(len(feature_columns)):
		axs[l, c].plot(range(len(r2_scores[idx_feature][idx])), r2_scores[idx_feature][idx])
		axs[l, c].set_title('Model (' + reg_stra + ') ' + str(idx_feature))
		if l == lines - 1:
			l = 0
			c += 1
		else:
			l += 1

	if display:
		plt.show()
	if save:
		plt.savefig('images/plot_r2/' + reg_stra + '.png', dpi=300)

	plt.close()

def plot_highest_target(y_sorted, best_query, reg_stra, iteration, display = False, save = True):
	check_images_dir('plot_highest_target/' + reg_stra)
	
	plt.figure()
	plt.scatter(range(len(y_sorted)), y_sorted, color = 'Black')
	plt.scatter(best_query, y_sorted[best_query], color = 'Red')

	if display:
		plt.show()
	if save:
		plt.savefig('images/plot_highest_target/' + reg_stra + '/iteration_' + str(iteration + 1) + '.png', dpi=300)

	plt.close()

