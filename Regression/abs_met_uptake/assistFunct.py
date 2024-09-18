import numpy as np
import os

def random_training_set(X, y, n_init):
	# Return a random training set of n_init elements
	idx_init = np.random.choice(range(len(X)), size = n_init, replace = False)
	X_train, y_train = X[idx_init], y[idx_init]
	return X_train, y_train, idx_init

def delete_data(X, y, query):
	# Avoids repetition
	X = np.delete(X, query, axis = 0)
	y = np.delete(y, query, axis = 0)
	return X, y

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
		maxVoteCount = -1
		idx_maxVoteCount = np.array([])
		for candidate in count:
			if candidate[1] > maxVoteCount:
				maxVoteCount = candidate[1]
				idx_maxVoteCount = [candidate[0]]

			elif candidate[1] == maxVoteCount:
				np.append(idx_maxVoteCount, candidate[0])

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

def new_bool_repr(boolRepr, query):
	# Update the boolean representation of which data is used in training (alProces.class_set[2])
	# Used because of =/= between absolute idx and testing idx.
	# Can (maybe) be optimized by rewriting all the alProcesses to not delete data from X_test/y_test but instead only use boolRepr

	nb_false = 0
	for idx in range(len(boolRepr)):
		if not boolRepr[idx]: # Virtually create a sub array containing all False values explored by nb_false
			found = False
			idx_query = 0
			while not found and idx_query < len(query):
				if nb_false == query[idx_query]: # If the False value is queried
					boolRepr[idx] = True
					found = True
				idx_query += 1
			nb_false += 1

	return boolRepr

def check_images_dir(dir):
	if len(dir) > 0:
		dir_split = dir.split('/')
		path = './'
		for idx_dir in range(len(dir_split)):
			path += dir_split[idx_dir] + '/'
			os.makedirs(path, exist_ok = True)


