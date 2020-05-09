"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p2_ransac.py

Purpose: do RANSAC algorithm on a list of points
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import sys


def is_inlier(coord, a, b, c, tau):
    """
    check if a point is an inlier to a line with tolerance of tau
    :param coord: the point
    :param a: cos(theta) of the line
    :param b: sin(theta) of the line
    :param c: -rho of the line
    :param tau: tolerance level
    :return: true if the point is an inlier. False otherwise.
    """
    return (np.dot(coord, (a, b)) + c) ** 2 < tau ** 2


def is_outlier(coord, a, b, c, tau):
    """
    check if a point is an outlierlier to a line with tolerance of tau
    :param coord: the point
    :param a: cos(theta) of the line
    :param b: sin(theta) of the line
    :param c: -rho of the line
    :param tau: tolerance level
    :return: true if the point is an outlier. False otherwise.
    """
    return (np.dot(coord, (a, b)) + c) ** 2 > tau ** 2


def dis_point_line(points, a, b, c):
    """
    Calculate the distance from each point to the line.
    :param points: list of points
    :param a: cos(theta) of the line
    :param b: sin(theta) of the line
    :param c: -rho of the line
    :return: list of distance
    """
    dis = np.abs(np.dot(points, (a, b)) + c)
    dis = dis / np.sqrt((a ** 2 + b ** 2))
    return dis


if __name__ == "__main__":
    """
    Handle the command-line arguments
    """
    seed = None
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Usage: %s points.txt samples tau [seed]\n" % sys.argv[0])
        sys.exit()
    elif os.path.isfile(sys.argv[1]):
        point_file = sys.argv[1]
        n_samples = int(sys.argv[2])
        tau = float(sys.argv[3])
        if len(sys.argv) == 5:
            seed = int(sys.argv[4])
    else:
        print("File not found: ", sys.argv[1])
        sys.exit()

    # read the list of coords
    points = np.loadtxt(point_file)

    # apply seed if declared
    if seed is not None:
        np.random.seed(seed)

    """
    RANSAC
    """
    k_max = 0
    m_theta = m_rho = 0
    for i in range(n_samples):
        # random select 2 different points as sample
        sample = np.random.randint(0, points.shape[0], 2)
        if sample[0] == sample[1]:
            continue

        # calculate the line through the 2 points
        p0, p1 = points[sample]
        dx, dy = p1 - p0
        theta = np.arctan2(dy, dx) + np.pi / 2
        if theta > np.pi:
            theta -= np.pi
        a, b = np.cos(theta), np.sin(theta)
        rho = np.dot((a, b), p1)
        c = -rho

        # calculate the number of inlier
        k = points[is_inlier(points, a, b, c, tau)].shape[0]

        if k > k_max:
            k_max = k
            m_theta, m_rho = theta, rho
            print('Sample {:d}:'.format(i))
            print('indices ({:d},{:d})'.format(*sample))
            print('line ({:.3f},{:.3f},{:.3f})'.format(a, b, c))
            print('inliers {:d}'.format(k))
            print()

    a, b = np.cos(m_theta), np.sin(m_theta)
    c = -m_rho
    inliers = points[is_inlier(points, a, b, c, tau)]
    outliers = points[is_outlier(points, a, b, c, tau)]
    avg_inlier_dis = np.average(dis_point_line(inliers, a, b, c))
    avg_outlier_dis = np.average(dis_point_line(outliers, a, b, c))
    print('avg inlier dist {:.3f}'.format(avg_inlier_dis))
    print('avg outlier dist {:.3f}'.format(avg_outlier_dis))
