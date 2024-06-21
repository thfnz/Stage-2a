from sklearn.ensemble import RandomForestRegressor

def regression_strategy(reg_stra):
	match reg_stra:

		case 'randomForest':
			model = RandomForestRegressor(n_estimators = 200, criterion = 'squared_error', n_jobs = -1)

	return model