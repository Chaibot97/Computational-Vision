"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:   hw5_kmeans.py

Purpose: Apply k-means to images
"""

import cv2
import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt

import os
import sys


if __name__ == "__main__":
    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 3:
        print("Usage: %s image k\n" % sys.argv[0])
        sys.exit()

    img_name = sys.argv[1]
    k = int(sys.argv[2])

    # read in image
    im = cv2.imread(img_name)
    img_name, img_type = img_name.split('/')[-1].split('.')[-2:]

    # concatenate features with pixel position and RGB values
    x = np.arange(im.shape[1])/im.shape[1]*100
    y = np.arange(im.shape[0])/im.shape[0]*100
    x, y = np.meshgrid(x, y)
    all_values = np.concatenate(
        (x[:, :, np.newaxis], y[:, :, np.newaxis], im),
        axis = -1
    )
    all_values = all_values.reshape(-1, 5).astype(np.float32)

    # apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    num_reinitializations = 10
    initialization_method = cv2.KMEANS_PP_CENTERS
    ret, label, center = cv2.kmeans(all_values, k, None, criteria,
                                    num_reinitializations, initialization_method)

    # save the result image
    out_name = img_name + '_seg.' + img_type
    center = np.uint8(center)
    out = center[label.flatten()][:, 2:]
    out = out.reshape(im.shape)
    cv2.imwrite(out_name, out)
