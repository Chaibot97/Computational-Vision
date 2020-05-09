"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p1_block.py

Purpose: Compute a “downsized image” where each pixel represents
the average intensity across a region of the input image
and generate the block image by replacing each pixel in
the downsized image with a block of pixels having the same intensity.
"""

import cv2
import numpy as np

import os
import sys


def downsize(img, m, n):
    """
    Get the downsized version of a image
    :param img: original img
    :param m: number of rows of downsized img
    :param n: number of cols of downsized img
    :return: downsized img with dtype=np.float64
    """
    M, N = img.shape
    sm = M / m
    sn = N / n

    downsized_image = np.zeros((m, n), np.float64)

    for i in range(m):
        for j in range(n):
            sliced_img = img[int(i * sm):int((i + 1) * sm), int(j * sn):int((j + 1) * sn)]
            downsized_image[i, j] = np.average(sliced_img)

    return downsized_image


def binary(img):
    """
    Convert an image into binary image
    :param img: original img
    :return: (threshold, binary img with only black and white)
    """
    threshold = np.median(img)
    binary_img = np.copy(img)
    binary_img[img >= threshold] = 255
    binary_img[img < threshold] = 0
    return threshold, binary_img


def upsample(img, b):
    """
    Upsample image to block image
    :param img: original img
    :param b: scale
    :return: block image
    """
    block_image = np.repeat(img, b, axis=0)  # repeat along row axis
    block_image = np.repeat(block_image, b, axis=1)  # repeat along col axis
    return block_image


if __name__ == "__main__":
    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 5:
        print("Usage: %s image_dir m n b\n" % sys.argv[0])
        sys.exit()
    elif os.path.isfile(sys.argv[1]):
        img_name = sys.argv[1]
        m = int(sys.argv[2])
        n = int(sys.argv[3])
        b = int(sys.argv[4])
    else:
        print("Image not found: ", sys.argv[1])
        sys.exit()

    """
    Open the image using OpenCV
    """
    img = cv2.imread(img_name, 0)   # Load img in grey scale

    """
    Generate the downsized image
    """
    downsized_image = downsize(img, m, n)

    """
    Generate the block images
    """
    threshold, binary_image = binary(downsized_image)   # Generate the binary downsized image
    binary_image.astype(np.uint8)
    binary_block_image = upsample(binary_image, b) # Generate the binary block image

    downsized_image.astype(np.uint8)
    avg_block_image = upsample(downsized_image, b)  # Generate the average block image

    """
    Output results
    """
    print('Downsized images are (%d, %d)' % downsized_image.shape)
    print('Block images are (%d, %d)' % avg_block_image.shape)
    print('Average intensity at (%d, %d) is %.2f' % (m//3, n//3, downsized_image[m//3, n//3]))
    print('Average intensity at (%d, %d) is %.2f' % (m//3, 2*n//3, downsized_image[m//3, 2*n//3]))
    print('Average intensity at (%d, %d) is %.2f' % (2*m//3, n//3, downsized_image[2*m//3, n//3]))
    print('Average intensity at (%d, %d) is %.2f' % (2*m//3, 2*n//3, downsized_image[2*m//3, 2*n//3]))
    print("Binary threshold: %.2f" % threshold)

    tmp_name, tmp_format = img_name.split('.')
    block_image_name = tmp_name + "_g." + tmp_format
    cv2.imwrite(block_image_name, avg_block_image)
    print("Wrote image %s" % block_image_name.split('/')[-1])

    block_image_name = tmp_name + "_b." + tmp_format
    cv2.imwrite(block_image_name, binary_block_image)
    print("Wrote image %s" % block_image_name.split('/')[-1])