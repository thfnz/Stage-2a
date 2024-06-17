from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import Ridge

def regression_strategy(polyDeg, alpha, reg_stra):
	# Returns a model of the chosen regression strategy.

	match reg_stra:

		case 'polynom':
			model = make_pipeline(StandardScaler(), PolynomialFeatures(polyDeg), Ridge(alpha = alpha))

		case 'gradBoost':
			params = {
			'n_estimators' : 500,
			'max_depth' : 4,
			'min_samples_split' : 5,
			'learning_rate' : 0.01,
			'loss' : 'squared_error'
			} 
			model = GradientBoostingRegressor(**params)

		case 'randomForest':
			model = RandomForestRegressor(n_estimators = 1000, criterion = 'squared_error', n_jobs = -1)

	return model