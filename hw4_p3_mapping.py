"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p3_mapping.py

Purpose: map a image from one camera
        to another image by a camera with different angle
"""

import cv2
import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt

import os
import sys


def rotation_matrix(r):
    """
    calculate the rotation matrix according to the angles
    :param r: 3 dimensional angle
    :return: the rotation matrix
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r[0]), -np.sin(r[0])],
                   [0, np.sin(r[0]), np.cos(r[0])]])
    Ry = np.array([[np.cos(r[1]), 0, np.sin(r[1])],
                   [0, 1, 0],
                   [-np.sin(r[1]), 0, np.cos(r[1])]])
    Rz = np.array([[np.cos(r[2]), -np.sin(r[2]), 0],
                   [np.sin(r[2]), np.cos(r[2]), 0],
                   [0, 0, 1]])
    return np.dot(np.dot(Rx, Ry), Rz)


def count_overlaps(H,m,n,N):
    """
    calculate the overlaps and percentage
    of overlapping sample points
    before and after transformation
    :param H: the transformation matrix
    :param m: row dimension
    :param n: col dimension
    :return: number and percentage of overlapping pixels
    """
    # generate the sample points u1 on the original image
    delta_r = m / N / 2
    delta_c = n / N / 2
    r_space = np.linspace(delta_r, m - delta_r, N)
    c_space = np.linspace(delta_c, n - delta_c, N)
    r, c = np.meshgrid(r_space, c_space)
    r, c = np.ravel(r), np.ravel(c)
    u1 = np.vstack((c, r, np.ones_like(r)))

    # transform to u2 and normalize
    u2 = H.dot(u1)
    u2 /= u2[-1]

    # count the number of overlapping samples
    overlaps = np.array(np.where(
        (u2[0] <= n) & (u2[0] >= 0) & (u2[1] <= m) & (u2[1] >= 0)
    ))
    count = overlaps.shape[-1]
    percentage = count / N ** 2

    return count, percentage


if __name__ == "__main__":
    # Handle the command-line arguments
    if len(sys.argv) != 2:
        print("Usage: %s params.txt \n" % sys.argv[0])
        sys.exit()

    params = sys.argv[1]

    np.set_printoptions(precision = 3, suppress = True)

    # define the dimension of the image
    m, n = 4000, 6000

    # read the camera parameters
    with open(params) as file:
        r1 = np.radians(np.array(file.readline().split()).astype(np.float64))
        s1, ic1, jc1 = np.array(file.readline().split()).astype(np.float64)
        r2 = np.radians(np.array(file.readline().split()).astype(np.float64))
        s2, ic2, jc2 = np.array(file.readline().split()).astype(np.float64)
        N = int(file.readline().strip())

    # calculate the Rotation Matrix
    R1 = rotation_matrix(r1)
    R2 = rotation_matrix(r2)

    # calculate the Intrinsic Parameter Matrix
    K1 = np.array([[s1, 0, jc1], [0, s1, ic1], [0, 0, 1]])
    K2 = np.array([[s2, 0, jc2], [0, s2, ic2], [0, 0, 1]])

    # calculate H
    H = K2.dot(R2).dot(R1.T).dot(LA.inv(K1))
    H = H / LA.norm(H) * 1000
    if H[-1, -1] < 0:
        H = -H

    print('Matrix: H_21')
    print('{:.3f}, {:.3f}, {:.3f}'.format(*H[0]))
    print('{:.3f}, {:.3f}, {:.3f}'.format(*H[1]))
    print('{:.3f}, {:.3f}, {:.3f}'.format(*H[2]))

    # transform the 4 corners of image1
    u1 = np.array([[0, 0, 1], [n, 0, 1], [0, m, 1], [n, m, 1]])
    u2 = H.dot(u1.T)
    u2 /= u2[-1]
    upperLeft = np.min(u2[0:2, :], axis = 1)
    lowerRight = np.max(u2[0:2, :], axis = 1)
    print("Upper left: {:.1f} {:.1f}".format(upperLeft[1], upperLeft[0]))
    print("Lower right: {:.1f} {:.1f}".format(lowerRight[1], lowerRight[0]))

    # calculate overlapping samples from image1 to image2
    count, percentage = count_overlaps(H,m,n,N)
    print("H_21 overlap count {:d}".format(count))
    print("H_21 overlap fraction {:.3f}".format(percentage))

    # calculate overlapping samples from image2 to image1
    count, percentage = count_overlaps(LA.inv(H), m, n,N)
    print("H_12 overlap count {:d}".format(count))
    print("H_12 overlap fraction {:.3f}".format(percentage))

    # calculate the center direction of image2
    u = np.array([[n//2,m//2,1]]).T
    d2 = R2.T.dot(LA.inv(K2)).dot(u)
    d2 = np.ravel(d2)
    if d2[-1] < 0:
        d2 = -d2
    print("Image 2 center direction: ({:.3f}, {:.3f}, {:.3f})".format(*d2))
    