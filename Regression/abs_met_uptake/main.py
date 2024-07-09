import pandas as pd

from oracleOnly import oracleOnly
from baseline import randomQuery

from plotResults import plotResults
from comparisonAlProcessBaseline import comparisonAlProcessBaseline

# Loading dataset
dataset = pd.read_csv('./properties.csv')

feature_columns = ['dimensions', ' supercell volume [A^3]', ' density [kg/m^3]', ' surface area [m^2/g]',
' num carbon', ' num hydrogen', ' num nitrogen', ' num oxygen', ' num sulfur', ' num silicon', 
' vertices', ' edges', ' genus', 
' largest included sphere diameter [A]', ' largest free sphere diameter [A]', ' largest included sphere along free sphere path diameter [A]']

X = dataset[feature_columns].values # 69840 instances
y = dataset[' absolute methane uptake high P [v STP/v]'].values

# Model selection (randomForest - elasticNet - elasticNetCV - XGB - SVR)
reg_stra = ['XGB', 'randomForest']

# AL
nb_iterations = 130
batch_size = 1
batch_size_highest_value = 0
nb_members = 2
n_init = 10

alProcess = oracleOnly(nb_iterations, batch_size, batch_size_highest_value)
baseline = randomQuery(nb_iterations, batch_size + batch_size_highest_value)
comp = comparisonAlProcessBaseline(alProcess, baseline, X, y, reg_stra, nb_members, n_init)
comp.comparison_top_n_accuracy(10, display = False, save = True, pbar = True)