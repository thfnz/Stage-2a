import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from assistFunct import *
from regressors import regression_strategy

# Loading dataset
dataset = pd.read_csv('../properties.csv')

feature_columns = ['dimensions', ' supercell volume [A^3]', ' density [kg/m^3]', ' surface area [m^2/g]',
' num carbon', ' num hydrogen', ' num nitrogen', ' num oxygen', ' num sulfur', ' num silicon', 
' vertices', ' edges', ' genus', 
' largest included sphere diameter [A]', ' largest free sphere diameter [A]', ' largest included sphere along free sphere path diameter [A]']

X = dataset[feature_columns].values # 69840 instances
y = dataset[' absolute methane uptake high P [v STP/v]'].values

reg_stra = ['XGB']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Models
nb_members = 10
members_sets = []

for idx_model in range(nb_members):
	members_sets.append(reg_stra[0]) # Dirty way, need to change if different models are giving better results

# Training and predicting
for idx_model in range(nb_members):
	model = regression_strategy(members_sets[idx_model][0])
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	members_sets.append(y_pred)
	members_sets.append(r2_score(y_test, y_pred))

# y_pred_avg
y_pred_avg = []
for i in range(len(X_test[:, 0]))
	somme = 0
	for idx_model in range(nb_members):
		somme += members_sets[idx_model][1]
	y_pred_avg.append(somme / nb_members)

# plot_r2_avg
r2_avg = r2_score(y_test, y_pred_avg)
plt.figure()
plt.plot(range(len(r2_avg)), r2_avg)
plt.ylabel(r2_avg)
plt.show()

# plot_r2
plot_r2(members_sets, 2, 0, 0, reg_stra, lines = 2, columns = 5, display = True, save = False)