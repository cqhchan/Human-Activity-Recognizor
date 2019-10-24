# spot check on engineered-features
# https://machinelearningmastery.com/evaluate-machine-learning-algorithms-for-human-activity-recognition/
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
import numpy as np
filterET = [False, False, False, True, False, False, True, False, False, True, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, True, False, False, True, True, True, False, False, False, False, False
	, False, True, True, True, True, True, True, False, True, True, True, False
	, False, False, False, False, False, True, True, False, False, True, True, True
	, False, True, True, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, True, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, True, False, False, True, False
	, False, False, False, False, False, False, False, False, False, False, False, True
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, True, True, False
	, False, False, False, False, False, False, False, False, False, False, True, True
	, False, False, False, False, False, False, False, False, False, False, True, False
	, True, False, False, True, False, True, True, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, True, False, False, True, False, False, True, False, False, True, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, True, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, True, False, False, False, False, True, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, True, True
	, True, False, False, True, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, True, True, True];

filterDT = [False, False, False, False, False, False, False, False, False, True, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, True, False, False, False, False, False, False, False, False, False, False
	, False, True, True, False, True, True, True, True, False, True, False, False
	, False, False, False, False, False, False, False, True, False, False, False, True
	, False, False, True, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, True, False, False, False, False, False, False
	, False, False, False, False, False, True, False, False, False, False, False, False
	, False, False, False, True, False, False, False, False, False, True, False, False
	, False, True, False, False, False, False, False, False, False, True, False, False
	, True, False, False, False, False, False, False, True, True, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, True, False, True, False, False, False, False, False, False, False, False
	, True, True, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, True, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, True, False, False, False, False, False
	, False, False, False, False, False, False, False, False, True, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, True
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, True, False, False, False, False, False, False, False, False, False
	, False, False, False, False, True, False, False, False, False, False, False, False
	, False, False, False, False, True, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, True, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, True, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, True, False, False, False, False, False, True, False, False, True, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, True, False, False, False, False, False, False, False, False, False, False, True
	, False, False, False, False, True, False, True, False, False, False, False, False
	, False, False, False, False, False, True, False, False, True, False, False, False
	, False, False, False, False, False, False, True, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, True
	, False, False, False, False, True, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, False, False, False, False, False, False, True, False
	, False, False, False, False, False, False, False, False, False, False, False, False
	, False, False, False, False, True, False, True, True, False]

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	# load input data
	X = load_file(prefix + group + '/X_'+group+'.txt')
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	print(testX.shape, testy.shape)
	# flatten y
	trainy, testy = trainy[:,0], testy[:,0]
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
	# nonlinear models
	#models['knn'] = KNeighborsClassifier(n_neighbors=7)
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = LinearSVC()
	#models['bayes'] = GaussianNB()
	# ensemble models
	#models['bag'] = BaggingClassifier(n_estimators=100)
	#models['rf'] = RandomForestClassifier(n_estimators=100)
	models['et'] = ExtraTreesClassifier(n_estimators=100)
	#models['gbm'] = GradientBoostingClassifier(n_estimators=100)
	print('Defined %d models' % len(models))
	return models

# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model, name):
	# fit the model
	model.fit(trainX, trainy)
	# make predictions
	yhat = model.predict(testX)
	np.savetxt(name + ".txt",yhat)
	print(yhat)
	# evaluate predictions
	accuracy = accuracy_score(testy, yhat)
	return accuracy * 100.0

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
	results = dict()
	for name, model in models.items():
		# evaluate the model

		#print(len(filter))
		rfe = RFE(model, 50)
		rfe = rfe.fit(trainX, trainy)

		#print(rfe.support_);
		print(name);
		#newTrainX = trainX[:,rfe.support_]
		#newTestX = testX[:,rfe.support_]

		results[name] = evaluate_model(trainX, trainy, testX, testy, model,name)
		# show process

		print('>%s: %.3f' % (name, results[name]))
	return results

# print and plot the results
def summarize_results(results, maximize=True):
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,v) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	print()
	for name, score in mean_scores:
		print('Name=%s, Score=%.3f' % (name, score))

# load dataset
trainX, trainy, testX, testy = load_dataset()
# get model list
models = define_models()
# evaluate models
results = evaluate_models(trainX, trainy, testX, testy, models)
# summarize results
summarize_results(results)