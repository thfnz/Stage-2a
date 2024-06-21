from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

def regression_strategy(reg_stra):
	match reg_stra:

		case 'randomForest':
			model = RandomForestRegressor(n_estimators = 200, criterion = 'squared_error', n_jobs = -1)

		case 'elasticNet':
			model = ElasticNet() # CV to find best hyperparam ?

	return model