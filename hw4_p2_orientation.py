"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p2_orientation.py

Purpose: do keypoint gradient direction estimation
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import sys

if __name__ == "__main__":
    # Handle the command-line arguments
    if len(sys.argv) != 4:
        print("Usage: %s sigma img points\n" % sys.argv[0])
        sys.exit()

    sigma = float(sys.argv[1])
    im_name = sys.argv[2]
    pt_name = sys.argv[3]

    im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)

    #  Gaussian smoothing
    ksize = (int(4 * sigma + 1), int(4 * sigma + 1))
    im_s = cv2.GaussianBlur(im.astype(np.float64), ksize, sigma)

    #  Derivative kernels
    kx, ky = cv2.getDerivKernels(1, 1, 3)
    kx = np.transpose(kx / 2)
    ky = ky / 2

    #  Derivatives
    im_dx = cv2.filter2D(im_s, -1, kx)
    im_dy = cv2.filter2D(im_s, -1, ky)

    im_gd = np.arctan2(im_dy, im_dx) * 180 / np.pi
    im_gm = np.sqrt(im_dy ** 2 + im_dx ** 2)

    sigma_v = 2 * sigma
    width = int(2 * sigma_v)
    kp = np.loadtxt(pt_name, dtype=np.int)
    for i, p in enumerate(kp):
        print("\n Point {:d}: ({:d},{:d})".format(i, p[0], p[1]))

        # get the region for calculation
        im_gd_sub = im_gd[p[0] - width:p[0] + width + 1, p[1] - width:p[1] + width + 1]
        im_gm_sub = im_gm[p[0] - width:p[0] + width + 1, p[1] - width:p[1] + width + 1]

        length = int(2 * width + 1)
        center = length // 2

        # map all the angles from 0 to 360
        im_gd_sub = im_gd_sub + 180

        # weight of each pixel for voting
        weight = cv2.getGaussianKernel(length, sigma_v)
        weight = np.outer(weight, weight)
        weight = im_gm_sub * weight

        # calculate which bin each pixel belongs to and find the closest neighbour bin
        indices = (im_gd_sub // 10).astype(np.int)
        d_i = im_gd_sub / 10 - indices - 0.5
        nbr_i = ((indices + np.sign(d_i) + 36) % 36).astype(np.int)

        # linear interpolation
        vote = (1 - np.abs(d_i)) * weight
        vote_nbr = np.abs(d_i) * weight

        # generate histogram
        hist, _ = np.histogram(indices, bins=36, range=[0, 36], weights=vote)
        hist_nbr, _ = np.histogram(nbr_i, bins=36, range=[0, 36], weights=vote_nbr)
        hist += hist_nbr

        # apply smoothing to the histogram
        hist_r = np.concatenate((hist[1:], [hist[0]]))
        hist_l = np.concatenate(([hist[-1]], hist[:-1]))
        smoothed_hist = 0.5 * (hist + 0.5 * (hist_l + hist_r))

        print('Histograms:')
        for j in range(len(hist)):
            print("[{:d},{:d}]: {:.2f} {:.2f}".format(-180 + j * 10, -170 + j * 10, hist[j], smoothed_hist[j]))

        # find the local maximum in histogram
        smoothed_hist_r = np.concatenate((smoothed_hist[1:], [smoothed_hist[0]]))
        smoothed_hist_l = np.concatenate(([smoothed_hist[-1]], smoothed_hist[:-1]))
        peakpts = np.where((smoothed_hist > smoothed_hist_l) & (smoothed_hist > smoothed_hist_r))[0]

        # Apply parabolic interpolation to each peak where 'x' is theta in degree and 'y' is histogram value
        x = np.vstack((peakpts - 1, peakpts, peakpts + 1)).T
        x = x * 10 - 180 + 5
        y = np.vstack((smoothed_hist_l[peakpts], smoothed_hist[peakpts], smoothed_hist_r[peakpts])).T

        # calculate a,b,c of the parabolic equation
        a = (y[:, 2] - y[:, 1]) / (x[:, 2] - x[:, 1]) / (x[:, 2] - x[:, 0]) - (y[:, 0] - y[:, 1]) / (
                x[:, 0] - x[:, 1]) / (x[:, 2] - x[:, 0])
        b = (a * (x[:, 1] ** 2 - x[:, 0] ** 2) + (y[:, 0] - y[:, 1])) / (x[:, 0] - x[:, 1])
        c = y[:, 0] - a * x[:, 0] ** 2 - b * x[:, 0]

        # calculate the maximum point of the function
        peak_degree = -b / (2 * a)
        peak_value = c - b ** 2 / (4 * a)
        peak = np.vstack((peak_degree, peak_value)).T
        peak = np.array(sorted(peak, key=lambda p: p[1], reverse=True))
        for j, p in enumerate(peak):
            print("Peak {:d}: theta {:.1f}, value {:.2f}".format(j, *p))

        # count the number of peak that is within 0.8 of the maximum
        peak_value = peak[:, 1]
        maximum = peak_value[0]
        count = np.count_nonzero(peak_value[peak_value >= 0.8 * maximum])
        print("Number of strong orientation peaks: {:d}".format(count))
