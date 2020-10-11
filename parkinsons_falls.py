# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 10:48:03 2020

@author: SNIZAM
"""

########################################
######### Bio NLP Assignment 2 #########
########################################


import numpy as np
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import nltk
nltk.download('punkt')
from nltk import word_tokenize, sent_tokenize
from nltk.stem.porter import *
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time


stemmer = PorterStemmer()

word_clusters = {}

path ='/Users/sohailnizam/Desktop/Bio NLP/'


########## Helper Functions ##########

def preprocess_text(raw_text):
    #stemming and lowercasing (no stopword removal)
    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))

def loadwordclusters():
    infile = open(path + '/50mpaths2')
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getclusterfeatures(sent):
    sent = sent.lower()
    terms = nltk.word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)

def length_counter(text):
	wt = word_tokenize(text)

	return(len(wt))


########## Import/Preprocess Data ##########

#get the data
park_data = pd.read_csv(path + 'pdfalls.csv')

#clean up fall location categories
home_locations = ["home", "house", "bathroom", "bedroom", "kitchen", 
			 "living", "shower",
			 "indoor", "inside"]

fall_loc_bin = []
for index,row in park_data.iterrows():
	words = word_tokenize(row['fall_location'])
	at_home = 0
	for word in words:
		if word.lower() in home_locations or 'home' in word.lower():
			at_home = 1
			break
	fall_loc_bin.append(at_home)
		
park_data['at_home'] = fall_loc_bin
	
#binarize fall_class (1 = CoM, 0 = BoS or Other)
park_data['fall_class'] = pd.Series(np.where(park_data['fall_class'] == "CoM", 1, 0))

#change female to 1/0 coding
park_data['gender'] = pd.get_dummies(park_data['female'], drop_first = True)

#lowercase the fall_description text
park_data['fall_description'] = park_data['fall_description'].apply(lambda x: x.lower())

#add description length feature
park_data['description_length'] = park_data['fall_description'].apply(length_counter)


#train/test split
sss = StratifiedShuffleSplit(test_size = .20)

for train_index, test_index in sss.split(park_data, park_data['fall_class']):
	train = park_data.iloc[train_index]
	test = park_data.iloc[test_index]


#isolate the different features and outcome
X_train = train.drop(['fall_class', 'record_id', 'duration', 
					  'fall_location', 'fall_study_day', 'female'],
					 axis = 1)
X_test = test.drop(['fall_class', 'record_id', 'duration', 
					'fall_location', 'fall_study_day', 'female'],
				   axis = 1)

y_train = train['fall_class']
y_test = test['fall_class']



########## Pick the Best Classifier ##########


#The individual classifiers
gnb = GaussianNB()
kn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
svc = svm.SVC(gamma = 'scale', kernel = 'rbf')
rf = RandomForestClassifier()
ab = AdaBoostClassifier()

#the ensemble
ensemble = VotingClassifier(estimators=[('gnb',gnb), ('kn',kn), ('svc',svc),
										('dt', dt), ('rf',rf), ('ab',ab)],
							voting = 'hard')

#vectorizers and word clusters
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=100)
clustervectorizer = CountVectorizer(ngram_range=(1,1), max_features=100)
word_clusters = loadwordclusters()


#a function that vectorizes and creates the final feature df
def create_features(df):
	#isolate basic features
	age = np.array([df['age']]).T
	gender =  np.array([df['gender']]).T
	loc = np.array([df['at_home']]).T
	d_len = np.array([df['description_length']]).T
	
	#preprocess text for ngrams (not for clusters)
	text = df['fall_description'].apply(preprocess_text)
	ctext = df['fall_description'].apply(getclusterfeatures)

	#vectorize
	ngrams = vectorizer.fit_transform(text)
	clusters = clustervectorizer.fit_transform(ctext)

	#concatenate all the features
	ftrs = hstack((age, gender, loc, ngrams, clusters))
	#ftrs = np.concatenate((age, loc, gender, d_len), axis = 1)
	#ftrs = loc
	
	return(ftrs)



#create the pipelines for each classifier
ens_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(), 
									 accept_sparse = True)),
						 ('ens', ensemble)])

gnb_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(),
									 accept_sparse = True)),
						 ('gnb', gnb)])

kn_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(),
									 accept_sparse = True)),
						 ('kn', kn)])

svc_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(),
									 accept_sparse = True)),
						 ('svc', svc)])

rf_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(),
									 accept_sparse = True)),
						 ('rf', rf)])

ab_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(),
									 accept_sparse = True)),
						 ('ab', ab)])


#create the parameter grids
ens_grid = {'ens__kn__n_neighbors': [5, 10, 20],
			'ens__dt__max_depth' : [None, 10, 20],
		'ens__svc__C': [1, 10, 50, 100],
		'ens__rf__n_estimators': [10, 50, 100],
		'ens__ab__n_estimators': [10, 50, 100]}
kn_grid = {'kn__n_neighbors': [5, 10, 20]}
svc_grid = {'svc__C': [1, 10, 50, 100]}
rf_grid = {'rf__n_estimators': [10, 50, 100]}
ab_grid = {'ab__n_estimators': [10, 50, 100]}



#create the grid search
ens_grid_search = GridSearchCV(estimator = ens_pipe, param_grid = ens_grid, cv = 5, scoring = 'f1_micro')
kn_grid_search = GridSearchCV(estimator = kn_pipe, param_grid = kn_grid, cv = 5, scoring = 'f1_micro')
svc_grid_search = GridSearchCV(estimator = svc_pipe, param_grid = svc_grid, cv = 5, scoring = 'f1_micro')
rf_grid_search = GridSearchCV(estimator = rf_pipe, param_grid = rf_grid, cv = 5, scoring = 'f1_micro')
ab_grid_search = GridSearchCV(estimator = ab_pipe, param_grid = ab_grid, cv = 5, scoring = 'f1_micro')

#Do grid search for each model
t0 = time.time()
ens_grid_search.fit(X_train, y_train)
print("ensemble done")
kn_grid_search.fit(X_train, y_train)
print("knn done")
svc_grid_search.fit(X_train, y_train)
print("svm done")
rf_grid_search.fit(X_train, y_train)
print("random forest done")
ab_grid_search.fit(X_train, y_train)
print("adaboost done")
t1 = time.time()
print("Finished all in " + str((t1 - t0)/60) + " minutes.")

#identify the best parameter set for each
print(ens_grid_search.best_params_)
print(kn_grid_search.best_params_)
print(svc_grid_search.best_params_)
print(rf_grid_search.best_params_)
print(ab_grid_search.best_params_)

'''
{'ens__ab__n_estimators': 10, 'ens__dt__max_depth': 20, 
'ens__kn__n_neighbors': 20, 'ens__rf__n_estimators': 10, 'ens__svc__C': 1}

{'kn__n_neighbors': 5}
{'svc__C': 100}
{'rf__n_estimators': 50}
{'ab__n_estimators': 100}
'''

#Get cross validated f1_micro, f1_macro, and accuracy for each classifier with best params
gnb = GaussianNB()
kn = KNeighborsClassifier(n_neighbors=5)
kn2 = KNeighborsClassifier(n_neighbors=20)
dt = DecisionTreeClassifier(max_depth=20)
svc = svm.SVC(gamma = 'scale', kernel = 'rbf', C=100)
svc2 = svm.SVC(gamma = 'scale', kernel = 'rbf', C=1)
rf = RandomForestClassifier(n_estimators=50)
rf2 = RandomForestClassifier(n_estimators=10)
ab = AdaBoostClassifier(n_estimators=100)
ab2 = AdaBoostClassifier(n_estimators=10)

ensemble = VotingClassifier(estimators=[('gnb',gnb), ('kn',kn2), ('svc',svc2),
										('rf',rf2), ('ab',ab2), ('dt',dt)],
							voting = 'hard')


#create the pipelines for each classifier
ens_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(),
									 accept_sparse = True)),
						 ('ens', ensemble)])

gnb_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(),
									 accept_sparse = True)),
						 ('gnb', gnb)])

kn_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(),
									 accept_sparse = True)),
						 ('kn', kn)])

svc_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(),
									 accept_sparse = True)),
						 ('svc', svc)])

rf_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(),
									 accept_sparse = True)),
						 ('rf', rf)])

ab_pipe = Pipeline(steps = [('ftrs', FunctionTransformer(create_features)),
						 ('dense', FunctionTransformer(lambda x: x.todense(),
									 accept_sparse = True)),
						 ('ab', ab)])

#do the cv
ens_cv = cross_validate(ens_pipe, X_train, y_train, cv=5, scoring=['accuracy', 'f1_micro', 'f1_macro'])
gnb_cv = cross_validate(gnb_pipe, X_train, y_train, cv=5, scoring=['accuracy', 'f1_micro', 'f1_macro'])
kn_cv = cross_validate(kn_pipe, X_train, y_train, cv=5, scoring=['accuracy', 'f1_micro', 'f1_macro'])
svc_cv = cross_validate(svc_pipe, X_train, y_train, cv=5, scoring=['accuracy', 'f1_micro', 'f1_macro'])
rf_cv = cross_validate(rf_pipe, X_train, y_train, cv=5, scoring=['accuracy', 'f1_micro', 'f1_macro'])
ab_cv = cross_validate(ab_pipe, X_train, y_train, cv=5, scoring=['accuracy', 'f1_micro', 'f1_macro'])

cv_results = [ens_cv, gnb_cv, kn_cv, svc_cv, rf_cv, ab_cv]
for result in cv_results:
	print("f1_micro:" + str(np.mean(result['test_f1_micro'])))
	print("f1_macro:" + str(np.mean(result['test_f1_macro'])))
	print()


#results with description length
'''
f1_micro:0.7070175438596491
f1_macro:0.5606878306878308

f1_micro:0.6508771929824562
f1_macro:0.4537801706080349

f1_micro:0.7403508771929824
f1_macro:0.6511488639074846

f1_micro:0.6970760233918128
f1_macro:0.5002143853756758

f1_micro:0.619298245614035
f1_macro:0.4812962962962962

f1_micro:0.6421052631578947
f1_macro:0.49641202883138363
'''

#results without description length
'''
f1_micro:0.7619883040935673
f1_macro:0.6775517662414214

f1_micro:0.639766081871345
f1_macro:0.47234426806868407

f1_micro:0.7502923976608187
f1_macro:0.6802467902467904

f1_micro:0.6637426900584795
f1_macro:0.45717351289792896

f1_micro:0.6502923976608187
f1_macro:0.510114468864469

f1_micro:0.6982456140350877
f1_macro:0.5879108391608392
'''


########## Evaluate the Best Classifier (Ensemble) ##########

#first just get test prediciton scores
ens_pipe.fit(X_train, y_train)
ens_preds = ens_pipe.predict(X_test)
f1_score(ens_preds, y_test, average='micro') #.7917
f1_score(ens_preds, y_test, average='macro') #.7363
accuracy_score(ens_preds, y_test) #.7917


#next get train size vs test prediciton f1 scores
len_list = []
ens_accs = []

# for each subset of train set
for i in range(1, 11):
	# initialize the classifier
	svc_sub = svm.SVC(gamma='scale', C=1, kernel="rbf")
	gnb_sub = GaussianNB()
	rf_sub = RandomForestClassifier(n_estimators=10)
	ab_sub  =AdaBoostClassifier(n_estimators=10)
	dt_sub = DecisionTreeClassifier(max_depth=20)
	kn_sub = KNeighborsClassifier(n_neighbors=5)
	ens_sub = VotingClassifier(estimators=[('gnb', gnb_sub), ('svc', svc_sub), ('kn',kn_sub),
										   ('rf', rf_sub), ('ab',ab_sub), ('dt',dt_sub)], voting='hard')

	# set the pipelines to vectorize

	ens_sub_pipe = Pipeline(steps=[('ftrs', FunctionTransformer(create_features)),
							   ('dense', FunctionTransformer(lambda x: x.todense(),
															 accept_sparse=True)),
							   ('ens', ens_sub)])

	# set the proportion of the train set to be used
	prop = i / 10

	# fit the models
	X_train_sub = X_train[:int(prop * len(X_train))]
	y_train_sub = y_train[:int(prop * len(y_train))]
	ens_sub_pipe.fit(X_train_sub, y_train_sub)

	# predict on the full test set for each model
	ens_sub_preds = ens_sub_pipe.predict(X_test)

	# Get the accuracies, append to lists
	ens_accs.append(f1_score(ens_sub_preds, y_test, average='micro'))
	len_list.append(int(prop * len(X_train)))

	print(str(i) + "done.")

# store the results in a dataframe
train_f1mic_df = pd.DataFrame()
train_f1mic_df['train_size'] = len_list
train_f1mic_df['ens_acc'] = ens_accs

train_f1mic_df.to_csv("/Users/sohailnizam/Desktop/Bio NLP/training_f1mic_df", index=False)


#Finally, perform ablation study
#Just tweak the create_features fcn to subset features each time
# a function that vectorizes and creates the final feature df
def create_features(df):
	# isolate basic features
	age = np.array([df['age']]).T
	gender = np.array([df['gender']]).T
	loc = np.array([df['at_home']]).T
	d_len = np.array([df['description_length']]).T

	# preprocess text for ngrams (not for clusters)
	text = df['fall_description'].apply(preprocess_text)
	ctext = df['fall_description'].apply(getclusterfeatures)

	# vectorize
	ngrams = vectorizer.fit_transform(text)
	clusters = clustervectorizer.fit_transform(ctext)

	# concatenate all the features
	ftrs = hstack((age, loc, clusters, ngrams))
	# ftrs = np.concatenate((age, loc, gender, d_len), axis = 1)
	# ftrs = loc

	return (ftrs)

ens_pipe.fit(X_train, y_train)
ens_preds = ens_pipe.predict(X_test)
print(f1_score(ens_preds, y_test, average='micro'))
print(f1_score(ens_preds, y_test, average='macro'))
print(accuracy_score(ens_preds, y_test))

'''
Remove ngrams:
micro = .750
macro =  .667

Remove clusters:
micro = .792
macro = .736

Remove age:
micro = .667
macro = .625

Remove loc:
micro = .667
macro = .556

Remove gender:
micro = .750
macro = .697
'''

