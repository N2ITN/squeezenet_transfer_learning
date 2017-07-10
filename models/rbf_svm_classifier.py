import os

import sklearn
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib


def train_svm_classifier(features, labels, model_output_path='.'):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance

    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """

    X = features
    y = labels

    param = [{
        "kernel": ["linear"],
        "C": [1, 10, 100, 1000]
    }, {
        "kernel": ["rbf"],
        "C": [1, 10, 100, 1000],
        "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
    }]

    # request probability estimation
    svm = SVC(probability=True)
    svm.fit(X, y)

    print("\nBest parameters set:")
    print(svm.best_params_)

    y_predict = clf.predict(X_test)

    labels = sorted(list(set(labels)))
    print("\nConfusion matrix:")
    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(y_test, y_predict, labels=labels))

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))