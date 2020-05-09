"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p1_camera.py

Purpose: project points to image pixels
"""

import cv2
import numpy as np
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


def is_inside_img(coords, m, n):
    """
    determine if a point is inside the image
    :param coords: the coords of the point
    :param m: rows of img
    :param n: cols of img
    :return: corresponding string for inside and outside
    """
    if 0 <= coords[1] <= m and 0 <= coords[0] <= n:
        return 'inside'
    else:
        return 'outside'


if __name__ == "__main__":
    # Handle the command-line arguments
    if len(sys.argv) != 3:
        print("Usage: %s params.txt points.txt\n" % sys.argv[0])
        sys.exit()

    params = sys.argv[1]
    points = sys.argv[2]

    # define the dimension of the image
    m, n = 4000, 6000

    # read the camera parameters
    with open(params) as file:
        r = np.radians(np.array(file.readline().split()).astype(np.float64))
        t = np.array(file.readline().split()).astype(np.float64).reshape((3, 1))
        f, d, ic, jc = np.array(file.readline().split()).astype(np.float64)

    # read the points
    points = np.loadtxt(points,ndmin=2)

    # calculate the Rotation Matrix
    R = rotation_matrix(r)

    # calculate the Intrinsic Parameter Matrix
    s = f / (d / 1000)
    K = np.array([[s, 0, jc], [0, s, ic], [0, 0, 1]])

    # calculate M
    M = np.dot(K, np.concatenate((R.T, np.dot(-R.T, t)), axis=1))
    print('Matrix M:')
    print('{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(*M[0]))
    print('{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(*M[1]))
    print('{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(*M[2]))

    # 3d transform the points
    augmented_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    h_coords = np.dot(M, augmented_points.T)

    # convert back to img coordinates
    img_coords = h_coords.T[:, :-1] / h_coords.T[:, -1:]

    print('Projections:')
    for i in range(len(img_coords)):
        print('{:d}: {:.1f} {:.1f} {:.1f} => {:.1f} {:.1f} {:s}'.format(
            i, *points[i], *np.flip(img_coords[i]), is_inside_img(img_coords[i], m, n)))

    # visible if transformed point's w' > 0
    print('visible:', *np.where(h_coords[-1] > 0)[0])
    print('hidden:', *np.where(h_coords[-1] <= 0)[0])
