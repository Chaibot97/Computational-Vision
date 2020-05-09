"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p3_distance.py

Purpose: Find the two closest and two furthest images from the directory
according to the distance between two images’ average color vectors,
and the distance between their histograms.

"""

import cv2
import numpy as np

import os
import sys


def color_averages(img):
    """
    Calculate color average of an image on each channel
    :param img: the input image
    :return: color average of the image
    """
    return np.average(img, axis = (0, 1))


def histogram(img):
    """
    Calculate the histogram of an image on r, g, b
    :param img: the input image
    :return: frequency histogram of the image
    """
    BINS = 8
    RANGE = np.tile(np.array([0, 255]), (3, 1))

    # histogram of the first image
    r = np.ravel(img[:, :, 0])
    g = np.ravel(img[:, :, 1])
    b = np.ravel(img[:, :, 2])
    hist, endpoints = np.histogramdd([r, g, b], bins = BINS, range = RANGE)

    # normalize the images
    return hist/np.sum(hist)


def find_min_max(img_list):
    """
    Find the min and max distance values in a list of img
    :param img_list: List of image names
    :return: max and min distance pairs
    """
    features = []
    for img in img_list:
        img = cv2.imread(img)
        features.append((color_averages(img),histogram(img)))
    color_averages_min = (('', ''), np.infty)
    color_averages_max = (('', ''), 0)
    histogram_min = (('', ''), np.infty)
    histogram_max = (('', ''), 0)

    for ii in range(len(features)):
        for jj in range(ii+1, len(features)):
            # for color average
            color_averages_dis = np.linalg.norm((features[ii][0]-features[jj][0]),2)
            if color_averages_dis < color_averages_min[1]:
                color_averages_min = ((img_list[ii], img_list[jj]), color_averages_dis)
            if color_averages_dis > color_averages_max[1]:
                color_averages_max = ((img_list[ii], img_list[jj]), color_averages_dis)

            # for histogram
            histogram_dis = np.sqrt(np.sum(np.power(features[ii][1]-features[jj][1], 2)))
            if histogram_dis < histogram_min[1]:
                histogram_min = ((img_list[ii], img_list[jj]), histogram_dis)
            if histogram_dis > histogram_max[1]:
                histogram_max = ((img_list[ii], img_list[jj]), histogram_dis)

    return color_averages_min, color_averages_max, histogram_min, histogram_max


def print_results(min, max):
    """
    Print the result of the min, max pairs in specified format
    :param min: the min pair of images
    :param max: the max pair of images
    """
    print('Closest pair is (%s, %s)' % (min[0][0], min[0][1]))
    print('Minimum distance is %.3f' % min[1])
    print('Furthest pair is (%s, %s)' % (max[0][0], max[0][1]))
    print('Maximmum distance is %.3f' % max[1])


if __name__ == "__main__":
    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 2:
        print("Usage: %s image_folder\n" % sys.argv[0])
        sys.exit()
    elif os.path.isdir(sys.argv[1]):
        image_folder = sys.argv[1]
    else:
        print("Path not found: ", sys.argv[1])
        sys.exit()

    # load the list of image names in the folder
    os.chdir(image_folder)
    img_list = os.listdir('./')
    img_list = [name for name in img_list if 'jpg' in name.lower()]
    img_list.sort()

    c_min, c_max, h_min, h_max = find_min_max(img_list)
    # average color vectors distance
    print('Using distance between color averages.')
    print_results(c_min, c_max)

    print()

    # histogram distance
    print('Using distance between histograms.')
    print_results(h_min, h_max)
