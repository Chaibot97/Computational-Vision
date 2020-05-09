"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p1_shape.py

Purpose: compute and output the properties of a set of point coordinates in R2
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import sys


def correct_e_vector(v):
    """
    Resolve the ambiguity of an eigenvector
    by ensuring the x to be positive
    or when x is 0, y to be positive
    :param v: original eigenvector
    :return: corrected eigenvector
    """
    if v[0] < 0:
        return -v
    elif v[0] == 0:
        return np.abs(v)
    return v


if __name__ == "__main__":
    """
    Handle the command-line arguments
    """
    if os.path.isfile(sys.argv[1]):
        point_file = sys.argv[1]
        tau = float(sys.argv[2])
        outfig = sys.argv[3]
    else:
        print("File not found: ", sys.argv[1])
        sys.exit()

    # read the list of coords
    coords = np.loadtxt(point_file)

    # (a)
    print("min: ({:.3f},{:.3f})".format(*np.min(coords, axis = 0)))
    print("max: ({:.3f},{:.3f})".format(*np.max(coords, axis = 0)))

    # (b)
    avg = np.average(coords, axis = 0)
    print("com: ({:.3f},{:.3f})".format(*avg))

    # (c)
    M = np.cov(coords.T, bias = True)
    e_values, e_vectors = np.linalg.eig(M)
    e_values = np.sqrt(e_values)

    min_e = np.argmin(e_values)
    s_min = e_values[min_e]
    e_vectors_min = correct_e_vector(e_vectors[:, min_e])
    print('min axis: ({:.3f},{:.3f}), sd {:.3f}'.format(*e_vectors_min, s_min))

    # (d)
    max_e = np.argmax(e_values)
    s_max = e_values[max_e]
    e_vectors_max = correct_e_vector(e_vectors[:, max_e])
    print('max axis: ({:.3f},{:.3f}), sd {:.3f}'.format(*e_vectors_max, s_max))

    # （e）
    a, b = e_vectors_min[0], e_vectors_min[1]
    theta = np.arctan2(b, a)
    rho = np.dot([a, b], avg)
    print('closest point: rho {:.3f}, theta {:.3f}'.format(rho, theta))

    # (f)
    c = - rho
    print('implicit: a {:.3f}, b {:.3f}, c {:.3f}'.format(a, b, c))

    # (g)
    print('best as {:s}'.format('line' if s_min < tau * s_max else 'ellipse'))

    # plot
    x_min = np.floor(np.min(coords)) - 5
    x_max = np.ceil(np.max(coords)) + 5
    plt.axis([x_min, x_max, x_min, x_max])  # make the axis in equal scale

    x = np.linspace(x_min, x_max, 2)
    line = -(a / b) * x - (c / b)
    plt.plot(x, line, color = 'black', label = 'Best fit line')
    plt.scatter(coords[:, 0], coords[:, 1], color = 'blue', label = 'Coords')
    plt.scatter(*avg, color = 'red', label = 'Center of mass')

    plt.legend()
    plt.savefig(outfig)