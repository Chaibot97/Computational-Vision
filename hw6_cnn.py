"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:   hw6_cnn.py

Purpose: train and test a convolutional neural net work for image classification
"""

import cv2
import numpy as np
import pickle
import sklearn.svm
import sklearn.metrics
from sklearn.model_selection import cross_val_score
import numpy.linalg as LA
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import os
import sys


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # create two convolution layers
        self.conv1 = nn.Conv2d(3, 9, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(9, 15, 5)

        # Create two fully connected hidden layers
        self.fc1 = nn.Linear(15 * 13 * 13, N1, bias=True)
        self.fc2 = nn.Linear(N1, N2, bias=True)
        self.fc3 = nn.Linear(N2, Nout, bias=True)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 15 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def success_rate(pred_Y, Y):
    """
    calculate the success rate of the prediction
    :param pred_Y: predicted Y
    :param Y: true Y
    :return: the success rate
    """
    _, pred_Y_index = torch.max(pred_Y, 1)
    _, Y_index = torch.max(Y, 1)
    num_equal = torch.sum(pred_Y_index.data == Y_index.data).item()
    num_different = torch.sum(pred_Y_index.data != Y_index.data).item()
    rate = num_equal / float(num_equal + num_different)
    return rate


def confusion_matrix(pred_Y, Y):
    """
    produce the confusion matrix
    :param pred_Y: predicted Y
    :param Y: true Y
    :return: the confusion matrix
    """
    _, pred_Y_index = torch.max(pred_Y, 1)
    _, Y_index = torch.max(Y, 1)
    pred_Y_index = pred_Y_index.cpu().numpy()
    Y_index = Y_index.cpu().numpy()
    return sklearn.metrics.confusion_matrix(Y_index, pred_Y_index)


def convert_to_categories(Y):
    """
    convert boolean label into digits
    :param Y: boolean labels
    :return: digits labels
    """
    _, categories = torch.max(Y.data, 1)
    categories = torch.Tensor.long(categories)
    return Variable(categories)


def load_images(in_path):
    """
    load images from a folder into tensors
    :param in_path: the path to the image folder
    :return: classes of images, input, label
    """
    img_dirs = os.listdir(in_path)
    img_dirs = [name for name in img_dirs
                if os.path.isdir(os.path.join(in_path, name))]
    img_dirs.sort()

    imgs = []
    labels = []
    l = len(img_dirs)
    for i, dir in enumerate(img_dirs):
        img_dir = os.path.join(in_path, dir)
        img_list = os.listdir(img_dir)
        img_list = [name for name in img_list
                    if 'jpg' in name.lower()
                    or 'jpeg' in name.lower()
                    or 'png' in name.lower()]

        label = np.zeros(l)
        label[i] = 1

        for img_name in img_list:
            im_path = os.path.join(img_dir, img_name)
            img = cv2.imread(im_path)
            img = cv2.resize(img, (DIM, DIM))
            img = img.astype(np.float32)
            img = np.array([img[:, :, 0], img[:, :, 1], img[:, :, 2]])
            imgs.append(img)
            labels.append(label)

    x = torch.tensor(imgs)
    y = torch.tensor(labels)
    return img_dirs, x, y


def split_data(X, Y, n_valid):
    """
    split the data into train set and valid set
    :param X: inputs
    :param Y: labels
    :param n_valid: number of data in valid set
    :return: inputs and labels for train and valid set
    """
    indices = torch.randperm(X.shape[0])
    x_valid = X[indices[:n_valid]]
    y_valid = Y[indices[:n_valid]]
    x_train = X[indices[n_valid:]]
    y_train = Y[indices[n_valid:]]
    return x_train, y_train, x_valid, y_valid


def to_cuda(tensors):
    """
    copy the a list of tensors varibles to gpu if possible
    :param tensors: the list of tensors varibles
    :return: the list of tensors varibles on gpu
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)
    result = []
    for t in tensors:
        result.append(t.to(device))
    return result


if __name__ == "__main__":
    # width and height of the resized images
    DIM = 64

    # hyperparameter of the nn
    N1 = 100
    N2 = 100
    Nout = 5

    # size of the validation size
    n_valid = 200

    # Handle the command-line arguments
    if len(sys.argv) != 3:
        print("Usage: %s train_path test_path\n" % sys.argv[0])
        sys.exit()

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    # load data
    print("Loading images ...")
    classes, X_train, Y_train = load_images(train_path)
    _, X_test, Y_test = load_images(test_path)

    n_train = X_train.shape[0] - n_valid
    print(n_train, "images loaded for training")
    print(X_test.shape[0], "images loaded for testing")

    X_train, Y_train, X_valid, Y_valid = \
        split_data(X_train, Y_train, n_valid)

    # generate the cnn
    net = Net()
    net, X_train, Y_train, X_test, Y_test, X_valid, Y_valid = \
        to_cuda(
            [net, X_train, Y_train, X_test, Y_test, X_valid, Y_valid]
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)

    Y_test_c = convert_to_categories(Y_test)
    Y_train_c = convert_to_categories(Y_train)
    Y_valid_c = convert_to_categories(Y_valid)

    # initial validation loss
    pred_Y_valid = net(X_valid)
    valid_loss = criterion(pred_Y_valid, Y_valid_c)
    print('Initial validation loss: %.5f' % valid_loss.item())

    # train the cnn
    num_epochs = 300
    batch_size = 64
    n_batches = int(np.ceil(n_train / batch_size))
    for ep in range(num_epochs):
        #  Create a random permutation of the indices of the row vectors.
        indices = torch.randperm(n_train)

        #  Run through each mini-batch
        for b in range(n_batches):
            batch_indices = indices[b * batch_size:(b + 1) * batch_size]
            batch_X = X_train[batch_indices]
            batch_Y = Y_train_c[batch_indices]

            #  Run the network on each data instance in the minibatch
            #  and then compute the object function value
            pred_Y = net(batch_X)
            loss = criterion(pred_Y, batch_Y)

            #  do back propagation and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print loss every 10 epochs
        pred_Y = net(X_train)
        if ep != 0 and ep % 10 == 0:
            pred_Y_valid = net(X_valid)
            valid_loss = criterion(pred_Y_valid, Y_valid_c)
            print("Epoch %d loss: %.5f" % (ep, valid_loss.item()))
            print('Training success rate:', success_rate(pred_Y, Y_train))

    # print the result
    pred_Y_train = net(X_train)
    pred_Y_test = net(X_test)
    print()
    print('Training success rate:', success_rate(pred_Y_train, Y_train))
    print('Test success rate:', success_rate(pred_Y_test, Y_test))
    cm = confusion_matrix(pred_Y_test, Y_test)
    accuracy = np.diag(cm) / np.sum(cm, axis=1)
    for i, cls in enumerate(classes):
        print('\t{}: {:.3f}'.format(cls, accuracy[i]))
    print('confusion matrix:')
    print(cm)
