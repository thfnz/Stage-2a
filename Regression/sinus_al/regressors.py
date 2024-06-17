from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared

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

		case 'gaussian':
			noise_std=0
			expSine=1.0 * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
			rbfK= 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2))
			kernel = rbfK + WhiteKernel(1e-1)
			model = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2)

	return model