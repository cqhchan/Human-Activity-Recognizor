from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import sys
import re
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
	# nonlinear models
	models['knn'] = KNeighborsClassifier(n_neighbors=7)
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = LinearSVC()
	models['bayes'] = GaussianNB()
	models['logisticRegression'] =  LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
	# ensemble models
	models['bag'] = BaggingClassifier(n_estimators=100)
	models['rf'] = RandomForestClassifier(n_estimators=100)
	models['et'] = ExtraTreesClassifier(n_estimators=100)
	models['gbm'] = GradientBoostingClassifier(n_estimators=100)
	print('Defined %d models' % len(models))
	return models



def loadData(path, type):
    x = np.load(os.path.join(path, type,type +"_data.npy"), allow_pickle=True)
    y = np.load(os.path.join(path, type, type + "_label.npy"), allow_pickle=True)
    return x,y



models = define_models()



path = sys.argv[1]
train_data, train_label = loadData(path, "train")
test_data, test_label = loadData(path, "test")


for key in models:
    print(key)
    clf = models[key]

    print("fitting...")
    print(train_data.shape)
    clf.fit(train_data, train_label)

    print("predicting...")
    test_prediction = clf.predict(test_data)

    accuracy = accuracy_score(test_label, test_prediction)
    print(key)

    print(accuracy)
# # clf = SVC(kernel='linear', decision_function_shape='ovo')
# clf = ExtraTreesClassifier(n_estimators=100)



