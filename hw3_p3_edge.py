"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p3_edge.py

Purpose: find the edges in an image
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import sys


def E(img):
    """
    Move one pixel to the East on the image
    so that every pixel (i,j) maps to the (i,j+1) on the original img
    :param img: the original image
    :return: the image after moved
    """
    m, n = img.shape[0], img.shape[1]
    return np.c_[img[:, 1:], np.zeros((m, 1))]


def W(img):
    """
    Move one pixel to the West on the image
    so that every pixel (i,j) maps to the (i,j-1) on the original img
    :param img: the original image
    :return: the image after moved
    """
    m, n = img.shape[0], img.shape[1]
    return np.c_[np.zeros((m, 1)), img[:, :-1]]


def N(img):
    """
    Move one pixel to the North on the image
    so that every pixel (i,j) maps to the (i-1,j) on the original img
    :param img: the original image
    :return: the image after moved
    """
    m, n = img.shape[0], img.shape[1]
    return np.r_[np.zeros((1, n)), img[:-1, :]]


def S(img):
    """
    Move one pixel to the South on the image
    so that every pixel (i,j) maps to the (i+1,j) on the original img
    :param img: the original image
    :return: the image after moved
    """
    m, n = img.shape[0], img.shape[1]
    return np.r_[img[1:, :], np.zeros((1, n))]


def NE(img):
    """
    Move one pixel to the Northeast on the image
    so that every pixel (i,j) maps to the (i-1,j+1) on the original img
    :param img: the original image
    :return: the image after moved
    """
    return N(E(img))


def SW(img):
    """
    Move one pixel to the Southwest on the image
    so that every pixel (i,j) maps to the (i+1,j-1) on the original img
    :param img: the original image
    :return: the image after moved
    """
    return S(W(img))


def NW(img):
    """
    Move one pixel to the Northwest on the image
    so that every pixel (i,j) maps to the (i-1,j-1) on the original img
    :param img: the original image
    :return: the image after moved
    """
    return N(W(img))


def SE(img):
    """
    Move one pixel to the Southeast on the image
    so that every pixel (i,j) maps to the (i+1,j+1) on the original img
    :param img: the original image
    :return: the image after moved
    """
    return S(E(img))


if __name__ == "__main__":
    # Handle the command-line arguments
    if len(sys.argv) != 3:
        print("Usage: %s sigma in_img\n" % sys.argv[0])
        sys.exit()

    sigma = float(sys.argv[1])
    img_name = sys.argv[2]

    # Open the image using OpenCV
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_name, img_type = img_name.split('.')

    m, n = img.shape[0], img.shape[1]

    # apply Gaussian smoothing
    ksize = (int(4 * sigma + 1), int(4 * sigma + 1))
    im_s = cv2.GaussianBlur(img.astype(np.float32), ksize, sigma)

    # compute the gradient
    kx, ky = cv2.getDerivKernels(1, 1, 3)
    kx = np.transpose(kx / 2)
    ky = ky / 2
    im_dx = cv2.filter2D(im_s, -1, kx)
    im_dy = -cv2.filter2D(im_s, -1, ky)

    # gradient magnitude
    im_gm = np.sqrt(im_dx ** 2 + im_dy ** 2)
    
    # gradient angle
    im_gt = np.arctan2(im_dy, im_dx)

    # define colors in BGR corresponding to the directions and index
    #                       0               1              2           3          -1
    #                      W/E            NE/SW           N/S        NW/SE       empty
    #                      red            white           blue       green       black
    COLORS = np.array([(0, 0, 255), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)])

    # flip the negative angles
    im_gt = np.where(im_gt< 0, im_gt + np.pi, im_gt)

    # map the angles to directions in the way that
    # [0, pi/4) [pi/4, pi/2)    [pi/2, 3pi/4)    [3pi/4, pi)
    #   W/E        NE/SW            N/S             NW/SE
    im_gt = np.fmod((im_gt + np.pi / 8), np.pi)

    # map angels to indices from 0 to 4
    im_dir = (im_gt // (np.pi / 4)).astype(int)

    # remove the pixels with 0 magnitude
    im_dir = np.where((im_gm == 0), -1, im_dir)

    # remove the pixels on the boarder
    im_dir[:, 0] = im_dir[:, -1] = im_dir[0, :] = im_dir[-1, :] = -1

    # generate foo_dir image
    out_img_dir = np.where(im_gm < 1.0, -1, im_dir).astype(int)
    out_img_dir = COLORS[out_img_dir]

    # map the max grd to 255
    im_grd = 255 * im_gm / np.max(im_gm)

    # calculate the maximums by compare each pixels with the ones ahead and behind it
    # maximums are marked as 1 and others 0
    Maximum = np.where((im_dir == 0) & (im_gm >= W(im_gm)) & (im_gm >= E(im_gm)), 1, 0)
    Maximum += np.where((im_dir == 1) & (im_gm >= NE(im_gm)) & (im_gm >= SW(im_gm)), 1, 0)
    Maximum += np.where((im_dir == 2) & (im_gm >= N(im_gm)) & (im_gm >= S(im_gm)), 1, 0)
    Maximum += np.where((im_dir == 3) & (im_gm >= NW(im_gm)) & (im_gm >= SE(im_gm)), 1, 0)
    print('Number after non-maximum: {:d}'.format(np.sum(Maximum)))

    # apply the 1.0 threshold
    Maximum = np.where(im_gm < 1.0, 0, Maximum).astype(int)
    print('Number after 1.0 threshold: {:d}'.format(np.sum(Maximum)))

    # calculate stats of the remaining points
    remain_img = Maximum * im_gm
    remain_points = np.extract(remain_img != 0, remain_img)
    avg = np.average(remain_points)
    std = np.std(remain_points)
    threshold = min(avg + 0.5 * std, 30 / sigma)
    print('mu: {:.2f}\ns: {:.2f}\nThreshold: {:.2f}'.format(avg, std, threshold))

    # apply the threshold
    final_img = np.where(remain_img > threshold, remain_img, 0)
    final_img = 255 * final_img / np.max(final_img)
    final_points = np.extract(final_img != 0, final_img)
    print('Number after threshold: {:d}'.format(len(final_points)))

    # output the images
    cv2.imwrite(img_name + '_dir.' + img_type, out_img_dir)
    cv2.imwrite(img_name + '_grd.' + img_type, im_grd)
    cv2.imwrite(img_name + '_thr.' + img_type, final_img)
