import sys
import warnings
import pandas as pd

from oracleOnly import oracleOnly
from selfLabelingInde import selfLabelingInde
sys.path.append('./Tests')
from twoSteps import plot_top_n_accuracy, twoStepsNtop, twoStepsZone
from baseline import randomQuery, fastRandomQuery

from plotResults import plotResults
from assistPlot import assistPlot
from comparisonAlProcessBaseline import comparisonAlProcessBaseline
from comparisonAlProcessBaseline2steps import comparisonAlProcessBaseline2steps

# warnings.filterwarnings("ignore") # Skip convergence warnings

# Loading dataset
dataset = pd.read_csv('./properties.csv')

feature_columns = ['dimensions', ' supercell volume [A^3]', ' density [kg/m^3]', ' surface area [m^2/g]',
' num carbon', ' num hydrogen', ' num nitrogen', ' num oxygen', ' num sulfur', ' num silicon', 
' vertices', ' edges', ' genus', 
' largest included sphere diameter [A]', ' largest free sphere diameter [A]', ' largest included sphere along free sphere path diameter [A]']

X = dataset[feature_columns].values # 69840 instances
y = dataset[' absolute methane uptake high P [v STP/v]'].values

# Model selection (randomForest - [elasticNet, polynomialDegree, alpha] - elasticNetCV - XGB - SVR)
reg_stra = ['XGB', 'randomForest']
# reg_stra = ['XGB', 'randomForest', ['elasticNet', 3, 10], ['elasticNet', 4, 10]]

# AL
nb_members = 2
n_init = 5

nb_label_total = 150
nb_query_oracle = 100 # First step only
#nb_iterations = nb_query_oracle - nb_members * n_init
nb_iterations = 3

batch_size = 1
batch_size_highest_value = 0
batch_size_min_uncertainty = 0

threshold = 1e-3

# n_top = int(len(y) * 0.01)
n_top = 100

# alProcess = oracleOnly(nb_iterations, batch_size, batch_size_highest_value)
# alProcess = selfLabelingInde(threshold, nb_iterations, batch_size, batch_size_highest_value, batch_size_min_uncertainty, n_top)
# baseline = fastRandomQuery(nb_iterations, batch_size + batch_size_highest_value, n_top)

"""
# Evaluation of selected models
alProcess.initLearn(X, y, reg_stra, nb_members, n_init, display = False, pbar = True)

plot = plotResults(alProcess)
plot.top_n_accuracy(display = False, save = False)
plot.r2(2, 2, display = False, save = False)
plot.KDE_n_top(display = False, save = False)

astPlot = assistPlot(alProcess)
astPlot.self_labeled_data_amount(display = False, save = False)
"""

"""
# Comparison to a baseline
comp = comparisonAlProcessBaseline(alProcess, baseline, X, y, reg_stra, nb_members, n_init)
comp.comparison_top_n_accuracy(
	20, pbar = True,
	display_plot_top_n_accuracy = False, save_plot_top_n_accuracy = False, 
	display_plot_r2 = False, save_plot_r2 = False, 
	display_self_labeled_data_amount = False, save_self_labeled_data_amount = False,
	display = False, save = True)
"""

# """
# 2 steps tests
#n_top_train = nb_label_total - nb_query_oracle
n_top_train = 3
alProcess = selfLabelingInde(threshold, nb_iterations + n_top_train, batch_size, batch_size_highest_value, batch_size_min_uncertainty, n_top)
alProcess2steps = twoStepsNtop(threshold, nb_iterations, batch_size, batch_size_highest_value, batch_size_min_uncertainty, n_top, n_top_train)
baseline = fastRandomQuery(nb_iterations + n_top_train, batch_size + batch_size_highest_value, n_top)

# batch_size_scnd_step = 150 - nb_iterations
# alProcess = twoStepsZone(threshold, nb_iterations, batch_size_scnd_step, batch_size, batch_size_highest_value, batch_size_min_uncertainty, n_top_train = 100)

comp = comparisonAlProcessBaseline2steps(alProcess, alProcess2steps, baseline, X, y, reg_stra, nb_members, n_init)
comp.comparison_top_n_accuracy(
	20, pbar = True,
	display_plot_top_n_accuracy = False, save_plot_top_n_accuracy = True, 
	display_plot_r2 = False, save_plot_r2 = True, 
	display_self_labeled_data_amount = False, save_self_labeled_data_amount = feature_columns,
	display = False, save = True)
# """


