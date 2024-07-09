import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading dataset
dataset = pd.read_csv('properties.csv')

feature_columns = ['dimensions', ' supercell volume [A^3]', ' density [kg/m^3]', ' surface area [m^2/g]', 
' num carbon', ' num hydrogen', ' num nitrogen', ' num oxygen', ' num sulfur', ' num silicon', 
' vertices', ' edges', ' genus', 
' largest included sphere diameter [A]', ' largest free sphere diameter [A]', ' largest included sphere along free sphere path diameter [A]']

X = dataset[feature_columns] # 69840 instances
y = dataset[' absolute methane uptake high P [v STP/v]'].values

fig, axs = plt.subplots(4, 4)
ligne = 0
colonne = 0
for feature in feature_columns:
	axs[ligne, colonne].scatter(X[feature], y)
	axs[ligne, colonne].set_title(feature)
	if ligne == 3:
		ligne = 0
		colonne += 1
	else:
		ligne += 1
plt.show()