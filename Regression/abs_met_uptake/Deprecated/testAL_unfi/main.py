import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyprind
import sys
from sklearn.model_selection import train_test_split

from assistFunct import *
from query_strategies import *

# Loading dataset
dataset = pd.read_csv('../properties.csv')

feature_columns = ['dimensions', ' supercell volume [A^3]', ' density [kg/m^3]', ' surface area [m^2/g]', 
' num carbon', ' num hydrogen', ' num nitrogen', ' num oxygen', ' num sulfur', ' num silicon', 
' vertices', ' edges', ' genus', 
' largest included sphere diameter [A]', ' largest free sphere diameter [A]', ' largest included sphere along free sphere path diameter [A]']

X = dataset[feature_columns].values # 69840 instances
y = dataset[' absolute methane uptake high P [v STP/v]'].values

# Train/Test split
X_test, X_eval, y_test, y_eval = train_test_split(X, y, test_size = 0.1)

# Model selection
reg_stra_test = ['randomForest']

for reg_stra in reg_stra_test:
	print('Testing : ' + reg_stra)

	# Random training set
	n_init = 2
	X_train, y_train, X_test, y_test = random_training_set(X_test, y_test, n_init)

	# AL
	nb_iterations = 40
	batch_size = int(100 / 2)
	threshold = 1e-3
	r2_scores = []

	# Optional : pyprind progBar
	pbar = pyprind.ProgBar(nb_iterations, stream = sys.stdout)

	for iteration in range(nb_iterations):
		# Query strategy
		X_train, y_train, X_test, y_test, r2_score = uncertainty_sampling(X_train, y_train, X_test, y_test, X, y, reg_stra, batch_size, threshold, display = False)
		r2_scores.append(r2_score)

		# Optional : pyprind progBar
		pbar.update()

	# Results (final evaluation)
	predictor(X_train, y_train, X_eval, y_eval, reg_stra, X, y, display = True)
	plt.figure()
	plt.plot(range(len(r2_scores)), r2_scores)
	plt.show()