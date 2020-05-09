"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:   hw6_svm.py

Purpose: train a SVM with the training set
        and then test the accuracy of it
"""

import cv2
import numpy as np
import pickle
import sklearn.svm
import sklearn.metrics
from sklearn.model_selection import cross_val_score
import numpy.linalg as LA
from matplotlib import pyplot as plt

import os
import sys


def get_class(path):
    """
    read the classes of image from the text file
    :param path: the path to the file
    :return: list of classes
    """
    cls_dir = os.path.join(path, 'class.txt')
    with open(cls_dir, 'r') as f:
        classes = f.read().split('\n')
    return classes


def get_descriptor(path, classes):
    """
    read descriptors of all the classes from pickle files
    :param path: the directory containing the pickle files
    :param classes: the classes of descriptors needed
    :return: the list of descriptor arrays
    """
    descriptors = []
    for cls in classes:
        des_path = os.path.join(path, cls + '.pkl')
        with open(des_path, 'rb') as f:
            descriptors.append(pickle.load(f))
    return np.array(descriptors)


def prepare_data(descriptors):
    """
    combine the descriptors into a single array x
    and generate the corresponding y
    :param descriptors: list of descriptor arrays
    :return: descriptors and corresponding labels
    """
    y = np.empty(0)
    for i, des in enumerate(descriptors):
        label = (i + 1) * np.ones(des.shape[0])
        y = np.concatenate((y, label))
    x = np.vstack(descriptors)
    return x, y


def train_svm(cls, train_x, train_y):
    """
    train the binary SVM
    :param cls: the class that this SVM is about
    :param train_x: descriptors
    :param train_y: corresponding labels
    :return: the weight and bias of the SVM
    """
    svm = sklearn.svm.LinearSVC(max_iter=1000)
    svm.C = C[cls]

    svm.fit(train_x, train_y)
    error = 1 - svm.score(train_x, train_y)
    print('Training Error for "{}" : {:.4f}'.format(cls, error))
    w = svm.coef_
    b = svm.intercept_
    return w, b


def calculate_score(w,b,x):
    """
    calculate the normalized score of a SVM
    :param w: the weight of SVM
    :param b: the bias of SVM
    :param x: descriptors
    :return: the normalized score
    """
    score = x.dot(w.T)+b
    score /= LA.norm(w)
    return score


def predict(classifiers, x):
    """
    apply a list of SVMs to a list of inputs
    :param classifiers: weight and bias pairs
    :param x: input descriptor
    :return: result labels
    """
    scores = []
    for w, b in classifiers:
        scores.append(calculate_score(w, b, x))
    scores = np.array(scores)
    return np.argmax(scores, axis=0) + 1


def test_svm(classifiers, classes, test_x, test_y):
    """
    test the accuracy of a SVM
    print out the confusion matrix
    :param classifiers: weight and bias pairs
    :param classes: the class each classifier is for
    :param test_x: test inputs
    :param test_y: true labels
    """
    pred_y = predict(classifiers, test_x)
    cm = sklearn.metrics.confusion_matrix(test_y, pred_y)
    accuracy = np.diag(cm) / np.sum(cm, axis=1)
    print('Testing accuracy')
    for i, cls in enumerate(classes):
        print('\t{}: {:.3f}'.format(cls, accuracy[i]))
    total_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print('\tTotal: {:.3f}'.format(total_accuracy))
    print('confusion matrix:')
    print(cm)


if __name__ == "__main__":
    # best C found by cross validation with hw6_svm_cv.py
    C = {
            'grass': 2.3,
            'ocean': 1.5,
            'redcarpet': 0.1,
            'road': 6.2,
            'wheatfield': 2.4
        }

    # Handle the command-line arguments
    if len(sys.argv) != 3:
        print("Usage: %s train_path test_path\n" % sys.argv[0])
        sys.exit()

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    # get the categories of iamges
    classes = get_class(train_path)

    # read in descriptors
    train_des = get_descriptor(train_path, classes)
    test_des = get_descriptor(test_path, classes)

    # train SVM
    classifiers = []
    x, y = prepare_data(train_des)
    for i, cls in enumerate(classes):
        true_y = np.where(y == i+1, 1, -1)
        clf = train_svm(cls, x, true_y)
        classifiers.append(clf)

    # test SVM
    x, y = prepare_data(test_des)
    test_svm(classifiers, classes, x, y)


