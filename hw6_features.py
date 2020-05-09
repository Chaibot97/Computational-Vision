"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:   hw6_features.py

Purpose: extract the features from folders of images
        the features are the descriptor
        built from the 3D histograms of subimages
"""

import cv2
import numpy as np
import pickle
import numpy.linalg as LA
from matplotlib import pyplot as plt

import os
import sys


def hist3d(img):
    """
    Generate a 3D histogram from a image
    :param img: the image
    :return: 3D histogram
    """
    pixels = img.reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, (t, t, t))
    return hist


def generate_descriptor(img, bw, bh, t):
    """
    Generate descriptor for image
    using 3D histograms of subimages
    :param img: the image
    :return: descriptor of the size bw*bh*t^3
    """
    des = np.empty(0)
    H = img.shape[0]
    W = img.shape[1]
    dh = int(H / (bh + 1))
    dw = int(W / (bw + 1))

    # build the descriptor by
    # concatenating flattened 3D histograms of sub-images
    for n in range(bh):
        for m in range(bw):
            i = n * dh
            j = m * dw
            sub_img = img[i:i + 2 * dh, j:j + 2 * dw]
            hist = hist3d(sub_img)
            des = np.concatenate((des, hist.ravel()))
    return des


def gradient_hist(img, bw, bh, t):
    """
    Generate descriptor for image
    using gradient orientation histograms of subimages
    :param img: the image
    :return: descriptor of the size bw*bh*t^3
    """
    des = np.empty(0)
    H = img.shape[0]
    W = img.shape[1]
    #  Gaussian smoothing
    sigma = 2
    ksize = (int(4 * sigma + 1), int(4 * sigma + 1))
    im_s = cv2.GaussianBlur(img.astype(np.float64), ksize, sigma)

    #  Derivative kernels
    kx, ky = cv2.getDerivKernels(1, 1, 3)
    kx = np.transpose(kx / 2)
    ky = ky / 2

    #  Derivatives
    im_dx = cv2.filter2D(im_s, -1, kx)
    im_dy = cv2.filter2D(im_s, -1, ky)

    im_gd = np.arctan2(im_dy, im_dx) * 180 / np.pi

    dh = int(H / (bh + 1))
    dw = int(W / (bw + 1))

    # build the descriptor by
    # concatenating flattened gradient orientation of sub-images
    for n in range(bh):
        for m in range(bw):
            i = n * dh
            j = m * dw
            sub_img = im_gd[i:i + 2 * dh, j:j + 2 * dw]
            hist, _ = np.histogram(sub_img, bins = 36, range = [0, 36])
            des = np.concatenate((des, hist.ravel()))
    return des


if __name__ == "__main__":
    # Globe flag deciding do addition of gradient or not
    ADD_GRD = True

    # Handle the command-line arguments
    if len(sys.argv) != 3:
        print("Usage: %s in_path out_path\n" % sys.argv[0])
        sys.exit()

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Get images of different classes
    img_dirs = os.listdir(in_path)
    img_dirs = [name for name in img_dirs
                if os.path.isdir(os.path.join(in_path, name))]
    img_dirs.sort()

    # write the list of classes found to file
    out_name = os.path.join(out_path, 'class.txt')
    with open(out_name, 'w') as f:
        f.write('\n'.join(img_dirs))

    bw = bh = 4
    t = 4
    for dir in img_dirs:
        print(dir)
        img_dir = os.path.join(in_path, dir)
        img_list = os.listdir(img_dir)
        img_list = [name for name in img_list
                    if 'jpg' in name.lower()
                    or 'jpeg' in name.lower()
                    or 'png' in name.lower()]

        # build the descriptors
        descriptors = []
        for i, img_name in enumerate(img_list):
            im_path = os.path.join(img_dir, img_name)
            img = cv2.imread(im_path)
            des = generate_descriptor(img, bw, bh, t)

            # concatenate the gradient to the descriptor if needed
            if ADD_GRD:
                grd = gradient_hist(img, bw, bh, t)
                des = np.concatenate((des,grd))
            descriptors.append(des)
        descriptors = np.array(descriptors)

        # write the descriptors to pickle file
        out_name = os.path.join(out_path, dir + '.pkl')
        with open(out_name, 'wb') as f:
            pickle.dump(descriptors, f)
