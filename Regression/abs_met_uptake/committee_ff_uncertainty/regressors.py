from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet

def regression_strategy(reg_stra, alpha):
	match reg_stra:

		case 'randomForest':
			model = RandomForestRegressor(n_estimators = 200, criterion = 'squared_error', n_jobs = -1)

		case 'elasticNet':
			model = make_pipeline(PolynomialFeatures(), StandardScaler(), ElasticNet(alpha = alpha)) # CV to find best hyperparam ?

	return model