import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor

def regression_strategy(reg_stra):
	match reg_stra:

		case 'randomForest':
			model = RandomForestRegressor(n_estimators = 800, criterion = 'squared_error', n_jobs = -1)

		case ['elasticNet', int(degree), int(alpha)]:
			model = make_pipeline(PolynomialFeatures(degree = degree), StandardScaler(), ElasticNet(alpha = alpha))

		case 'elasticNetCV':
			cv = RepeatedKFold(n_splits = 10, n_repeats = 1, random_state = None)
			ratios = np.arange(1, 10, 1)
			alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
			model = make_pipeline(PolynomialFeatures(degree = 4), StandardScaler(), ElasticNetCV(l1_ratio = ratios, alphas = alphas, cv = cv, n_jobs = -1))

		case 'XGB':
			model = XGBRegressor(n_estimators = 800, max_depth = 5, eta = 0.02, subsample = 0.75, colsample_bytree = 0.3, reg_lambda = 0.6, reg_alpha = 0.15, nthread = -1)

		case 'SVR':
			model = make_pipeline(StandardScaler(), SVR())

		case 'catboost':
			model = CatBoostRegressor(depth = 6, learning_rate = 0.2, l2_leaf_reg = 2, iterations = 300, thread_count = -1, silent = True)

	return model