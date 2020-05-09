"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:     p4_checkerboard.py

Purpose: Given two images, not necessarily the same size,to create a
checkerboard image from reduced resolution versions of these images,
forms the “white” squares and the second image forms the “black” squares.

"""

import cv2
import numpy as np

import os
import sys

def crop_img(img):
    """
    Crop the center of a image to make it square
    :param img: the image need to be cropped
    :return: the cropped
    """
    M, N, _ = img.shape
    mi = min(M, N)
    cropped_img = np.zeros((mi,mi),dtype=img.dtype)
    cropped_info = [[0,0],[0,0]] # record the cropped region
    if M > N:
        # crop vertically
        start = (M - N) // 2
        cropped_img = img[start:start+N , :]
        cropped_info = [[start,0],[start+N-1, N-1]]
    elif M < N:
        # crop horizontally
        start = (N - M) // 2
        cropped_img = img[: , start:start + M]
        cropped_info = [[0,start], [M-1, start + M-1]]
    else:
        # not need to crop
        cropped_img = img
        cropped_info = [[0, 0], [M, N]]
    return cropped_img, cropped_info


if __name__ == "__main__":
    """
    Handle the command-line arguments
    """
    if len(sys.argv) != 7:
        print("Usage: %s img1 img2 out_img M N s\n" % sys.argv[0])
        sys.exit()
    elif os.path.isfile(sys.argv[1]):
        img1_name = sys.argv[1]
        img2_name = sys.argv[2]
        out_img = sys.argv[3]
        M = int(sys.argv[4])
        N = int(sys.argv[5])
        s = int(sys.argv[6])
    else:
        print("Image not found: ", sys.argv[1])
        sys.exit()

    """
    Open the image using OpenCV
    """
    img1 = cv2.imread(img1_name)
    img2 = cv2.imread(img2_name)

    # crop the first img
    cropped_img1, info = crop_img(img1)
    print("Image %s cropped at (%d,%d) and (%d,%d)" % (img1_name,info[0][0], info[0][1], info[1][0], info[1][1]))
    # resized the cropped version of the first img
    img1 = cv2.resize(cropped_img1, (s, s))
    s1 = cropped_img1.shape
    s2 = img1.shape
    print("Resized from (%d, %d, %d) to (%d, %d, %d)" % (s1[0], s1[1], s1[2], s2[0], s2[1], s2[2]))

    # crop the second img
    cropped_img2, info = crop_img(img2)
    print("Image %s cropped at (%d,%d) and (%d,%d)" % (img2_name, info[0][0], info[0][1], info[1][0], info[1][1]))
    # resized the cropped version of the second img
    img2 = cv2.resize(cropped_img2, (s, s))
    s1 = cropped_img2.shape
    s2 = img2.shape
    print("Resized from (%d, %d, %d) to (%d, %d, %d)" % (s1[0], s1[1], s1[2], s2[0], s2[1], s2[2]))

    """
    Generate the unit-board in the form:
                                        1 2
                                        2 1
    """
    concat_img12 = np.concatenate((img1,img2),axis=1)
    concat_img21 = np.concatenate((img2,img1),axis=1)
    concat_img = np.concatenate((concat_img12,concat_img21),axis=0)

    # repeat the unit board to make the full-sized board
    board = np.tile(concat_img,(M//2,N//2,1))

    print('The checkerboard with dimensions %d X %d was output to %s'% (board.shape[0],board.shape[1],out_img))
    cv2.imwrite(out_img, board)