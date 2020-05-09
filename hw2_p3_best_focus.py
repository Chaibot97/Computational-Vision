"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p3_best_focus.py

Purpose: Determine which image is focused the best.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

import os
import sys


if __name__ == "__main__":
    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 2:
        print("Usage: %s image_dir\n" % sys.argv[0])
        sys.exit()
    elif os.path.isdir(sys.argv[1]):
        img_dir = sys.argv[1]
    else:
        print("Dir not found: ", sys.argv[1])
        sys.exit()

    # load the list of image names in the folder
    os.chdir(img_dir)
    img_list = os.listdir('./')
    img_list = [name for name in img_list if 'jpg' in name.lower()]
    img_list.sort()

    # Calculate gradients
    max_E = 0
    m_img = ""
    for img_name in img_list:
        img = cv2.imread(img_name, 0)
        im_dx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        im_dy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

        pixels = img.shape[0]*img.shape[1]
        E = np.sum(im_dx**2 + im_dy**2)
        E /= pixels

        if E>max_E:
            max_E=E
            m_img = img_name
        print('{:s}: {:.2f}'.format(img_name,E))

    print('Image {:s} is best focused.'.format(m_img))