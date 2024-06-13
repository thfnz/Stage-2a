import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from itertools import combinations
from sklearn.feature_selection import SelectFromModel

# Missing data

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

dataset = pd.read_csv(StringIO(csv_data)) # Turns csv_data into file object
# print(dataset.isnull().sum())

dataset.dropna() # Drops missing value cf https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html

# Impute missing values via the COLUMNS mean

imr = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imr.fit(dataset)
imputed_data = imr.transform(dataset)
# print(imputed_data)

dataset.fillna(dataset.mean(), inplace = True) # other way
# print(dataset)

# Qualitative variables : Nominal and ordinal (can be ordered) features

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']

# Mapping ordinal features

size_mapping = {'XL' : 3, 'L' : 2, 'M' : 1}

df['size'] = df['size'].map(size_mapping)
# print(df)

inv_size_mapping = {v: k for k, v in size_mapping.items()} # Dic backward
df['size'].map(inv_size_mapping)
# print(inv_size_mapping) # No difference, order in the tuples doesn't matter

# Encoding class labels

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)
# print(df)

# class_le = LabelEncoder()
# y = class_le.fit_transform(df['classlabel'].values)
# df['classlabel'] = y
# print(df)

# One-hot encoding on nominal features (all the legal combinaitions of values contain a single high)

X = df[['color', 'size', 'price']].values
# color_le = LabelEncoder()
# X[:, 0] = color_le.fit_transform(X[:, 0])
# print(X)

# X = df[['color', 'size', 'price']].values
# color_ohe = OneHotEncoder()
# a = color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

# c_transf = ColumnTransformer(transformers = [('onehot', OneHotEncoder(), [0])], remainder = 'passthrough')
# X = c_transf.fit_transform(X)
# print(X)

# print(pd.get_dummies(df[['price', 'color', 'size']])) # One-hot using bools
# print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)) # k dummies, if it's not k-1 dummies, it's obviously de k one (multicollinearity guard)

# c_transf = ColumnTransformer(transformers = [('oh', OneHotEncoder(categories='auto', drop='first'), [0])], remainder = 'passthrough')
# print(c_transf.fit_transform(X))

# Exemple : 

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
# print(df_wine.head())

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, stratify = y)

# Features onto the same scale

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

# Regularization :
# for C in np.arange(1., 30., 10):
	# lr = LogisticRegression(penalty = 'l1', C = C, solver = 'liblinear')
	# lr.fit(X_train_std, y_train)
	# print('\nValeur de c : ', C)
	# print('Training accuracy:', lr.score(X_train_std, y_train))
	# print('Test accuracy:', lr.score(X_test_std, y_test))

# fig = plt.figure()
# ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

# weights, params = [], []
# for c in np.arange(-4., 6.):
    # lr = LogisticRegression(penalty = 'l1', C = 10. ** c, solver = 'liblinear', random_state=0)
    # lr.fit(X_train_std, y_train)
    # weights.append(lr.coef_[1])
    # params.append(10**c)

# weights = np.array(weights)

# for column, color in zip(range(weights.shape[1]), colors):
    # plt.plot(params, weights[:, column],
             # label=df_wine.columns[column + 1],
             # color=color)
# plt.axhline(0, color='black', linestyle='--', linewidth=3)
# plt.xlim([10**(-5), 10**5])
# plt.ylabel('Weight coefficient')
# plt.xlabel('C (inverse regularization strength)')
# plt.xscale('log')
# plt.legend(loc='upper left')
# ax.legend(loc='upper center', 
          # bbox_to_anchor=(1.38, 1.03),
          # ncol=1, fancybox=True)
# plt.show()

# Sequential feature selection

class SBS: 
	# Greedy squential feature selection

	def __init__(self, estimator, k_features, test_size = 0.25, scoring = accuracy_score, random_state = 1):
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state
		self.scoring = scoring

	def transform(self, X):
		# returns only useful features
		return X[:, self.indices_]

	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		# fit + predict + calc score
		self.estimator.fit(X_train[:, indices], y_train)
		y_pred = self.estimator.predict(X_test[:, indices])
		score = self.scoring(y_test, y_pred)
		return score

	def fit(self, X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)
		
		# Ini + 1er eval
		dim = X_train.shape[1]
		self.indices_ = tuple(range(dim))
		self.subsets_ = [self.indices_]
		score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
		self.scores_ = [score]

		while dim > self.k_features:
			scores = []
			subsets = []

			for p in combinations(self.indices_, r = dim - 1): # Pour toutes les combinaisons de dim - 1 indices (removes 1 feature per iteration)
				# Eval du subset
				score = self._calc_score(X_train, y_train, X_test, y_test, p)
				scores.append(score)
				subsets.append(p)

			best = np.argmax(scores) # Best subset
			self.indices_ = subsets[best]
			self.subsets_.append(self.indices_)
			dim -= 1
			self.scores_.append(scores[best])

		self.k_score_ = self.scores_[-1]

		return self

knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
# k_feat = [len(k) for k in sbs.subsets_]

# plt.plot(k_feat, sbs.scores_, marker='o')
# plt.ylim([0.7, 1.02])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
# plt.show()

# k4 = list(sbs.subsets_[-4])
# print(df_wine.columns[1:][k4], '\n')

# knn.fit(X_train_std, y_train)
# print('Training accuracy (full dataset) :', knn.score(X_train_std, y_train))
# print('Test accuracy (full dataset) :', knn.score(X_test_std, y_test), '\n')

# knn.fit(X_train_std[:, k4], y_train)
# print('Training accuracy (4 features) :', knn.score(X_train_std[:, k4], y_train))
# print('Test accuracy (4 features) :', knn.score(X_test_std[:, k4], y_test))

# Assessing feature importance w/ Random Forests
labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators = 500, random_state = 1, n_jobs = -1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1] # descending indices

# for f in range(X_train.shape[1]):
# 	print(f + 1, labels[indices[f]], importances[indices[f]])

# plt.title('Feature importance')
# plt.bar(range(X_train.shape[1]), importances[indices], align='center')
# plt.xticks(range(X_train.shape[1]), labels[indices], rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# plt.show()

# print(sum(importances)) # = 1

sfm = SelectFromModel(estimator = forest, threshold = .1, prefit = True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold criterion:', X_selected.shape[1])

for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            labels[indices[f]], 
                            importances[indices[f]]))

