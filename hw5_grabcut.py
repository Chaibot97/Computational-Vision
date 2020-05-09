"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:   hw5_grabcut.py

Purpose: Apply GrabCut algorithm to an image
"""

import cv2
import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt

import os
import sys


def read_rect(file):
    """
    Load the rectangles from file
    in the form
        type topleft_x topleft_y bottomright_x bottomright_y
    where
        type 0 = outer box
        type 1 = inner box for exclusion
        type 2 = inner box for inclusion
    :param file: the txt file
    :return: rectangles of three types
    """
    rects = np.loadtxt(file, np.int)
    types = rects[:, 0]
    rects = rects[:, 1:]
    outer = rects[0]
    excluded = rects[types == 1, :]
    included = rects[types == 2, :]
    return outer, excluded, included


def draw_rects(img, rects, color, thick):
    """
    Draw rectangles on image
    :param img: the original image
    :param rects: rectangles
    :param color: color of the border
    :param thick: thickness of the border
    :return: the result image
    """
    for rect in rects:
        startpt = (rect[0], rect[1])
        endpt = (rect[2], rect[3])
        img = cv2.rectangle(img, startpt, endpt, color, thick)
    return img


if __name__ == "__main__":
    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 3:
        print("Usage: %s image rectangles\n" % sys.argv[0])
        sys.exit()

    img_name = sys.argv[1]
    rects = sys.argv[2]

    # read in rectangles and images
    outer, excluded, included = read_rect(rects)
    im_name, im_type = img_name.split('.')
    img = cv2.imread(img_name)

    # generate the image marked with rectangles
    marked_img = img.copy()
    marked_img = draw_rects(marked_img, [outer], (255, 0, 0), 4)
    marked_img = draw_rects(marked_img, excluded, (0, 255, 0), 2)
    marked_img = draw_rects(marked_img, included, (0, 0, 255), 2)
    out_name = im_name + '_marked.' + im_type
    cv2.imwrite(out_name, marked_img)

    # apply the outer rectangle
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = tuple(outer)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # apply the inner rectangles
    mask = draw_rects(mask, excluded, 0, -1)
    mask = draw_rects(mask, included, 1, -1)
    mask, bgdModel, fgdModel = cv2.grabCut(
        img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK
    )

    # save the result image
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask[:, :, np.newaxis]
    out_name = im_name + '_seg.' + im_type
    cv2.imwrite(out_name, img)
