import time
import numpy as np

from regressors import regression_strategy

def find_target_max_value(X_train, y_train, X_test, reg_stra, batch_size, display = False):
	#### TODO : If max query already in X_train --> end loop ?
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
	query = np.argsort(y_pred)[-batch_size:]

	return y_pred, query