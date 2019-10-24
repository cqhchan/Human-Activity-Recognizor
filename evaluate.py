from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
import re
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# create a dict of standard models to evaluate {name:object}



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax





def define_models(models=dict()):
    # nonlinear models
    #models['knn'] = KNeighborsClassifier(n_neighbors=7)
    #models['cart'] = DecisionTreeClassifier()
    #models['svmrbf'] = SVC(kernel='rbf', gamma='scale')
    #models['svmlinear'] = LinearSVC(C=8.0)
    #models['svmpoly'] = SVC(kernel="poly")
    #models['svmsigmoid'] = SVC(kernel='sigmoid')
    #models['bayes'] = GaussianNB()
    #models['logisticRegression'] =  LogisticRegression(C=1e5, solver='lbfgs',
    #multi_class='multinomial')
    # ensemble models
    #models['bag'] = BaggingClassifier(n_estimators=100)
    models['rf'] = RandomForestClassifier(n_estimators=100)
    models['et'] = ExtraTreesClassifier(n_estimators=200)
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
class_names = [0,1,2,3,4,5]

print(train_data.shape)
print(test_data.shape)

for key in models:
   # print(key)
    clf = models[key]
    #rfe = RFE(clf, train_data.shape[1])#svm linear 100
    #rfe = rfe.fit(train_data, train_label)

    #newTrainX = train_data[:,rfe.support_]
    #newTestX = test_data[:,rfe.support_]

  #  print("fitting...")
   # print(train_data.shape)
    clf.fit(train_data, train_label)

   # print("predicting...")
    test_prediction = clf.predict(test_data)

    accuracy = accuracy_score(test_label, test_prediction)
    print(key)

    print(accuracy)
    test_label = test_label.astype(int)
    test_prediction = test_prediction.astype(int)

    plot_confusion_matrix(test_label, test_prediction, classes=class_names, normalize=True,
                          title=key + " : " + str(accuracy))
    plt.show()

# # Plot non-normalized confusion matrix
# 	plot_confusion_matrix(test_label, test_prediction, classes=class_names,
# 					  title='Confusion matrix, without normalization')

# Plot normalized confusion matrix



# # clf = SVC(kernel='linear', decision_function_shape='ovo')
# clf = ExtraTreesClassifier(n_estimators=100)



