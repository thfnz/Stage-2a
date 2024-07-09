import pandas as pd

from oracleOnly import oracleOnly
from plotResults import plotResults

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

test = oracleOnly(nb_iterations, batch_size, batch_size_highest_value)
test.initLearn(X, y, reg_stra, nb_members, n_init, display = False, pbar = True)
plot = plotResults(test)
plot.top_n_accuracy(display = True, save = True)