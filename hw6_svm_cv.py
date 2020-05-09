"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:   hw6_svm_cv.py

Purpose: run Cross Validation on the training set
        to find the best C for each classes
"""

import cv2
import numpy as np
import pickle
import sklearn.svm
from sklearn.model_selection import cross_val_score
import numpy.linalg as LA
from matplotlib import pyplot as plt

import os
import sys


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


if __name__ == "__main__":
    # Handle the command-line arguments
    if len(sys.argv) != 2:
        print("Usage: %s train_path\n" % sys.argv[0])
        sys.exit()

    train_path = sys.argv[1]

    cls_dir = os.path.join(train_path, 'class.txt')
    with open(cls_dir, 'r') as f:
        classes = f.read().split('\n')

    # get the categories of iamges
    descriptors = []
    for cls in classes:
        des_path = os.path.join(train_path, cls + '.pkl')
        with open(des_path, 'rb') as f:
            descriptors.append(pickle.load(f))

    # read in descriptors
    descriptors = np.array(descriptors)

    # run cross validation
    C_s = np.arange(0.1, 10.1, 0.1)
    x, y = prepare_data(descriptors)
    for i, cls in enumerate(classes):
        true_y = np.where(y == i + 1, 1, -1)

        svm = sklearn.svm.LinearSVC(max_iter=10000)

        scores = []
        for C in C_s:
            svm.C = C
            this_scores = cross_val_score(svm, x, true_y, cv=5, n_jobs=4)
            scores.append(np.mean(this_scores))

        scores = np.array(scores)
        C = C_s[np.argmax(scores)]
        print("{} {:.1f}".format(cls, C))

