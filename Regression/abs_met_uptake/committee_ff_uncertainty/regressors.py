import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import RepeatedKFold

def regression_strategy(reg_stra, alpha):
	match reg_stra:

		case 'randomForest':
			model = RandomForestRegressor(n_estimators = 200, criterion = 'squared_error', n_jobs = -1)

		case 'elasticNet':
			model = make_pipeline(PolynomialFeatures(), StandardScaler(), ElasticNet(alpha = alpha))

		case 'elasticNetCV':
			cv = RepeatedKFold(n_splits = 5, n_repeats = 1, random_state = None)
			ratios = np.arange(1, 10, 1)
			alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
			model = make_pipeline(PolynomialFeatures(), StandardScaler(), ElasticNetCV(l1_ratio = ratios, alphas = alphas, cv = cv, n_jobs = -1))

	return model