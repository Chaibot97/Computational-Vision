"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p2_seam_carve.py

Purpose: do Seam Carving for Content-Aware Image Resizing
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import sys

if __name__ == "__main__":
    # Handle the command-line arguments
    if len(sys.argv) != 2:
        print("Usage: %s img\n" % sys.argv[0])
        sys.exit()

    img_name = sys.argv[1]

    # Open the image using OpenCV
    img = cv2.imread(img_name).astype(np.float32)
    img_name, img_type = img_name.split('.')

    M, N, channels = img.shape
    vertical = True
    # if it is a portrait image, transpose it
    if M > N:
        img = np.transpose(img, (1, 0, 2))
        vertical = False
        M, N, channels = img.shape

    # number of seams needed
    n_seams = N - M

    for seam in range(n_seams):
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im_dx = cv2.Sobel(grey_img, cv2.CV_32F, 1, 0)
        im_dy = cv2.Sobel(grey_img, cv2.CV_32F, 0, 1)

        # update weights
        W = np.abs(im_dx) + np.abs(im_dy)
        
        # make the energy on the left and right edge infinity
        W[:, 0] = np.inf
        W[:, -1] = np.inf
        for i in range(1, M):
            left = W[i - 1, :-2]
            right = W[i - 1, 2:]
            center = W[i - 1, 1:-1]
            W[i, 1:-1] += np.min((left, right, center), axis = 0)

        # calculate the seam vector c(·)
        c = np.zeros(M, dtype = np.uint16)
        c[-1] = np.argmin(W[-1])
        for i in range(M - 2, -1, -1):
            last_j = c[i + 1]
            c[i] = np.argmin(W[i, last_j - 1:last_j + 2]) + last_j - 1

        # generate the carved image
        carved_img = np.empty((M, N - 1 - seam, channels), dtype = np.float32)
        for i in range(M):
            j = c[i]
            carved_img[i] = np.concatenate((img[i, 0:j], img[i, j + 1:]))
            if seam == 0:
                img[i, j] = (0, 0, 255)

        # output the first seam img
        if seam == 0:
            if vertical:
                cv2.imwrite(img_name + '_seam.' + img_type, img)
            else:
                cv2.imwrite(img_name + '_seam.' + img_type, np.transpose(img, (1, 0, 2)))

        img = carved_img

        # print the coords of seams
        if seam in [0, 1, n_seams - 1]:
            print('\nPoints on seam {:d}:'.format(seam))
            if vertical:
                print('vertical')
                print('{:d}, {:d}'.format(0, c[0]))
                print('{:d}, {:d}'.format(M // 2, c[M // 2]))
                print('{:d}, {:d}'.format(M - 1, c[-1]))
            else:
                print('horizontal')
                print('{:d}, {:d}'.format(c[0], 0))
                print('{:d}, {:d}'.format(c[M // 2], M // 2))
                print('{:d}, {:d}'.format(c[-1], M - 1))
            print('Energy of seam {:d}: {:.2f}'.format(seam, W[-1, c[-1]] / M))

    # output the final img
    if vertical:
        cv2.imwrite(img_name + '_final.' + img_type, img)
    else:
        cv2.imwrite(img_name + '_final.' + img_type, np.transpose(img, (1, 0, 2)))
