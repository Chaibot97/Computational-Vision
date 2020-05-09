"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p2_shade.py

Purpose: Output the input image and its shaded version side-by-side in a single image file

"""

import cv2
import numpy as np

import os
import sys


def generate_mask(mask_dir, M, N):
    """
    Generate the normalized 2D mask according to the dir and size specified.
    Each pixel should be mapped from 1 to 0 according to the distance from the starting point.
    :param mask_dir: the side or corner of the image where the shading starts
    :param M: dimension of col vectors
    :param N: dimension of row vectors
    :return: The normalized 2D mask
    """
    if mask_dir == 'center':
        r_center = M // 2
        c_center = N // 2

        """
        Generate col vector
        """
        c1 = np.arange(r_center, 0, step=-1)
        c2 = np.arange(0, M - r_center)
        c = np.concatenate((c1, c2)).reshape(M, 1)

        """
        Generate row vector
        """
        r1 = np.arange(c_center, 0, step=-1)
        r2 = np.arange(0, N - c_center)
        r = np.concatenate((r1, r2))

        dir_map = np.sqrt(r ** 2 + c ** 2)
    elif mask_dir == 'left':
        r = np.arange(N)    # Generate row vector
        dir_map = np.tile(r, (M, 1))
    elif mask_dir == 'right':
        r = np.arange(N)    # enerate row vector
        r = np.flip(r)
        dir_map = np.tile(r, (M, 1))
    elif mask_dir == 'top':
        c = np.arange(M)    # Generate col vector
        dir_map = np.tile(c, (N, 1)).T
    elif mask_dir == 'bottom':
        c = np.arange(M)    # Generate col vector
        c = np.flip(c)
        dir_map = np.tile(c, (N, 1)).T
    else:
        print("ERROR: dir must be on of below:\n left, top, right, bottom, center")
        sys.exit()

    # return the normalized mask
    return 1 - dir_map / np.max(dir_map)


def print_mask_results(mask):
    """
    Print the values of mask on important points
    :param mask: the 2D mask
    """
    M, N = mask.shape
    rows = (0, M//2, M-1)
    cols = (0, N//2, N-1)
    for row in rows:
        for col in cols:
            print("(%d,%d) %.3f" % (row, col, mask[row, col]))


if __name__ == "__main__":
    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 4:
        print("Usage: %s in_img out_img dir\n" % sys.argv[0])
        sys.exit()
    elif os.path.isfile(sys.argv[1]):
        in_img = sys.argv[1]
        out_img = sys.argv[2]
        mask_dir = sys.argv[3]
    else:
        print("Image not found: ", sys.argv[1])
        sys.exit()


    """
    Open the image using OpenCV
    """
    img = cv2.imread(in_img, 1)  # Load img in RGB

    np.set_printoptions(precision=3)

    M, N, _ = img.shape

    mask = generate_mask(mask_dir, M, N)
    print_mask_results(mask)

    # apply the mask to each channel of the original image
    shaded_img = mask.reshape((M, N, 1)) * img
    shaded_img.astype(img.dtype)

    # concatenate the original and shaded images
    combined_img = np.concatenate((img, shaded_img), axis=1)

    cv2.imwrite(out_img, combined_img)
