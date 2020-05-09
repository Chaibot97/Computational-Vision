"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p1_compare.py

Purpose: Compare the result of Harris Measure and Orb
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import sys


def harris_measure(im_name, sigma):
    """
    Apply Harris Measure to a image
    :param im_name: the file name of the input image
    :param sigma: sigma for Gaussian smoothing
    :return: the list of keypoints
    """
    im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)

    #  Gaussian smoothing
    ksize = (4 * sigma + 1, 4 * sigma + 1)
    im_s = cv2.GaussianBlur(im.astype(np.float32), ksize, sigma)

    #  Derivative kernels
    kx, ky = cv2.getDerivKernels(1, 1, 3)
    kx = np.transpose(kx / 2)
    ky = ky / 2

    #  Derivatives
    im_dx = cv2.filter2D(im_s, -1, kx)
    im_dy = cv2.filter2D(im_s, -1, ky)

    # Components of the outer product
    im_dx_sq = im_dx * im_dx
    im_dy_sq = im_dy * im_dy
    im_dx_dy = im_dx * im_dy

    # Convolution of the outer product with the Gaussian kernel
    h_sigma = 2 * sigma
    h_ksize = (4 * h_sigma + 1, 4 * h_sigma + 1)
    im_dx_sq = cv2.GaussianBlur(im_dx_sq, h_ksize, h_sigma)
    im_dy_sq = cv2.GaussianBlur(im_dy_sq, h_ksize, h_sigma)
    im_dx_dy = cv2.GaussianBlur(im_dx_dy, h_ksize, h_sigma)

    # Compute the Harris measure
    kappa = 0.004
    im_det = im_dx_sq * im_dy_sq - im_dx_dy * im_dx_dy
    im_trace = im_dx_sq + im_dy_sq
    im_harris = im_det - kappa * im_trace * im_trace

    # Renormalize the intensities into the 0..255 range
    i_min = np.min(im_harris)
    i_max = np.max(im_harris)
    # print("Before normalization the minimum and maximum harris measures are",
    #       i_min, i_max)
    im_harris = 255 * (im_harris - i_min) / (i_max - i_min)

    # Apply non-maximum thresholding using dilation
    max_dist = 2 * sigma
    kernel = np.ones((2 * max_dist + 1, 2 * max_dist + 1), np.uint8)
    im_harris_dilate = cv2.dilate(im_harris, kernel)
    im_harris[np.where(im_harris < im_harris_dilate)] = 0

    # Get the normalized Harris measures of the peaks
    indices = np.where(im_harris > 0)
    ys, xs = indices[0], indices[1]
    peak_values = im_harris[indices]

    # Put them into the keypoint list
    kp_size = 4 * sigma
    kp = [
        cv2.KeyPoint(xs[i], ys[i], kp_size, _response=peak_values[i])
        for i in range(len(xs))
    ]

    # sort the keypoints in descending order
    kp = sorted(kp, key=lambda p: p.response, reverse=True)
    print('\nTop 10 Harris keypoints:')
    for i, k in enumerate(kp[:10]):
        print('{:d}: ({:.1f}, {:.1f}) {:.4f}'.format(i, *k.pt, k.response))

    out_im = cv2.drawKeypoints(im.astype(np.uint8), kp[:200], None)
    im_name, im_type = im_name.split('.')
    out_im_name = im_name + '_harris.' + im_type
    cv2.imwrite(out_im_name, out_im)

    return kp


def orb(im_name):
    """
    Apply Orb to a image
    :param im_name: the file name of the input image
    :return: the list of keypoints
    """
    im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)

    num_features = 1000
    orb = cv2.ORB_create(num_features)
    kp, des = orb.detectAndCompute(im, None)

    # Get rid of the pts with size greater than 45
    kp = filter(lambda p: p.size <= 45, kp)

    # sort the keypoints in descending order
    kp = sorted(kp, key=lambda p: p.response, reverse=True)
    print('\nTop 10 ORB keypoints:')
    for i, k in enumerate(kp[:10]):
        print('{:d}: ({:.1f}, {:.1f}) {:.4f}'.format(i, *k.pt, k.response))

    out_im = cv2.drawKeypoints(im, kp[:200], None)
    im_name, im_type = im_name.split('.')
    out_im_name = im_name + '_orb.' + im_type
    cv2.imwrite(out_im_name, out_im)
    return kp


def print_distance(kp1, kp2):
    """
    Calculate the median and mean of both pixel and the ranking difference of two sets of keypoints
    :param kp1: first list of keypoints
    :param kp2: second list of keypoints
    """
    # get the coordinates of the points with high ranking
    size1 = min(len(kp1), 100)
    kp1 = np.array([p.pt for p in kp1[0:size1]])
    size2 = min(len(kp2), 200)
    kp2 = np.array([p.pt for p in kp2[0:size2]])

    # calculate both the pixel and the ranking difference of each pair of closest points
    img_dis = np.zeros(size1)
    rank_dis = np.zeros(size1)
    for i, p in enumerate(kp1):
        dis = np.sum(np.square(kp2 - p), axis=1)
        j = np.argmin(dis)
        img_dis[i] = np.sqrt(dis[j])
        rank_dis[i] = np.abs(i - j)

    print("Median distance: {:.1f}".format(np.median(img_dis)))
    print("Average distance: {:.1f}".format(np.average(img_dis)))
    print("Median index difference: {:.1f}".format(np.median(rank_dis)))
    print("Average index difference: {:.1f}".format(np.average(rank_dis)))


if __name__ == "__main__":
    # Handle the command-line arguments
    if len(sys.argv) != 3:
        print("Usage: %s sigma img\n" % sys.argv[0])
        sys.exit()

    sigma = int(sys.argv[1])
    im_name = sys.argv[2]

    kp_h = harris_measure(im_name, sigma)
    kp_o = orb(im_name)

    print("\nHarris keypoint to ORB distances:")
    print_distance(kp_h, kp_o)

    print("\nORB keypoint to Harris distances:")
    print_distance(kp_o, kp_h)
