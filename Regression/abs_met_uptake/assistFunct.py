import numpy as np

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

def new_datasets(X_train, y_train, X_test, y_test, query):
	## TODO implement threshold
	# Labeling
	answer = ask_oracle(y_test, query)

	# New dataset
	X_train = np.concatenate((X_train, X_test[query]), axis = 0)
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