import pyprind
import sys
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# Opening DataFrame

pbar = pyprind.ProgBar(50000, stream = sys.stdout)

basepath = 'aclImdb'
labels = {'pos' : 1, 'neg' : 0}
df = pd.DataFrame()
for s in ('test', 'train'):
	for l in ('pos', 'neg'):
		path = os.path.join(basepath, s, l) # Search in each diff folders
		for file in sorted(os.listdir(path)): # Select all files
			with open(os.path.join(path, file), 'r', encoding = 'utf-8') as content:
				txt = content.read()
				df = df._append([[txt, labels[l]]], ignore_index = True) # Create dataset
				pbar.update()

df.columns = ['review', 'sentiment']

# Suffling Data

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

# Cleaning text data (removing <br /> etc)

def preprocessor(text):
	text = re.sub('<[^>]*>', '', text) # re.sub(pattern to be replaced, replacement, targeted string)
	emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
	text = (re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
	return text

df['review'] = df['review'].apply(preprocessor)

# Words into tokens

def tokenizer(text):
	return text.split()

porter = PorterStemmer() # Stemmer (cf notes)

def tokenizer_porter(text):
	return [porter.stem(word) for word in text.split()]

# Preping data

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents = None, lowercase = False, preprocessor = None, token_pattern = None) # term frequency inverse document frequency (cf notes)

lr_tfidf = Pipeline([('vectorization', tfidf), ('clf', LogisticRegression(solver = 'liblinear'))])

# Finding the best hyperparameters

stop = stopwords.words('english')

small_param_grid = [{'vectorization__ngram_range': [(1, 1)],
					'vectorization__stop_words': [None],
					'vectorization__tokenizer': [tokenizer, tokenizer_porter],
					'clf__penalty': ['l2'],
					'clf__C': [1.0, 10.0]},
                    {'vectorization__ngram_range': [(1, 1)],
					'vectorization__stop_words': [stop, None],
					'vectorization__tokenizer': [tokenizer],
					'vectorization__use_idf':[False],
					'vectorization__norm':[None],
					'clf__penalty': ['l2'],
					'clf__C': [1.0, 10.0]},
					]

gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid, scoring = 'accuracy', cv = 5, verbose = 1, n_jobs = -1)
gs_lr_tfidf.fit(X_train, y_train)

print(f'Best parameter set: {gs_lr_tfidf.best_params_}')
print(f'CV Accuracy: {gs_lr_tfidf.best_score_:.3f}')

clf = gs_lr_tfidf.best_estimator_
print(f'Test Accuracy: {clf.score(X_test, y_test):.3f}')

