"""
Author:   Lizhou Cai
Course:   CSCI 4270
File:   hw5_align.py

Purpose: Create mosaic of multiple images
"""

import cv2
import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt

import os
import sys


def create_mosaic(resImg, desImg, H, origin=(0, 0)):
    """
    Create mosaic of two images
    :param resImg: the images need to be transformed
    :param desImg: the target image
    :param H: the Homography matrix
    :param origin: topleft position of the target image
    :return: mosaic iamge
    """
    # map the positions of corners of resImg to desImg
    corner = np.array(
        [[0, 0],
         [0, resImg.shape[0] - 1],
         [resImg.shape[1] - 1, resImg.shape[0]],
         [resImg.shape[1] - 1, 0]]
    ).reshape((-1, 1, 2))
    corner = np.int32(cv2.perspectiveTransform(corner.astype(np.float64), H))

    # filter out the pixels that is too far away
    corner_x = corner[:, 0, 0]
    corner_y = corner[:, 0, 1]
    corner_x = corner_x[np.abs(corner_x) < resImg.shape[1] + desImg.shape[1]]
    corner_y = corner_y[np.abs(corner_y) < resImg.shape[0] + desImg.shape[0]]

    # calculate the topleft and bottomright positions of the final mosaic
    tl = [np.min(corner_x), np.min(corner_y)]
    br = [np.max(corner_x), np.max(corner_y)]

    x_offset = abs(min(tl[0], origin[0]) - origin[0])
    y_offset = abs(min(tl[1], origin[1]) - origin[1])

    tl = [min(tl[0], origin[0]), min(tl[1], origin[1])]
    br = [max(br[0], desImg.shape[1]), max(br[1], desImg.shape[0])]

    # calculate the shape of the final mosaic
    final_shape = np.array([br[1] - tl[1], br[0] - tl[0], 3])

    # calculate the translation matrix the to match the topleft point
    translation = np.array([
        [1, 0, np.abs(tl[0])],
        [0, 1, np.abs(tl[1])],
        [0, 0, 1]])

    # create the first part of the mosaic form resImg
    slice1 = cv2.warpPerspective(resImg,
                                 translation.dot(H),
                                 (final_shape[1], final_shape[0]),
                                 flags = cv2.INTER_NEAREST)

    # create the second part of the mosaic form desImg
    slice2 = np.zeros(final_shape)
    slice2[
    y_offset:y_offset + desImg.shape[0],
    x_offset:x_offset + desImg.shape[1]
    ] = desImg

    # create the averaging blending mask
    mask1 = np.where(slice2 == [0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5])
    mask2 = np.where(slice1 == [0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5])

    # blend the two parts to get the final mosaic
    mosaic = slice1 * mask1 + slice2 * mask2
    return mosaic, tl


def create_full_mosaic(anchor, others, img_dir):
    """
    Create mosaic out of an list of images
    :param anchor: name of the image that is going to be the center
    :param others: name of the other images and their H matrices to anchor
    :param img_dir: the directory to the images
    :return: the final mosaic, names of image used, file type of the image
    """
    name, type = anchor.split('.')
    names = [name]
    anchor = cv2.imread(os.path.join(img_dir, anchor))
    origin = [0, 0]

    # create the mosaic by combining other images with the anchor image
    for img in others:
        names.append(img.split('.')[0])
        H = others[img]
        img = cv2.imread(os.path.join(img_dir, img))
        anchor, origin = create_mosaic(img, anchor, H, origin)

    return anchor, names, type


def find_anchor(img_pairs, H_matrices):
    """
    Find the best anchor image by
    finding the image with the most number of neighbours
    :param img_pairs: pairs of images
    :param H_matrices: H matrices of each pairs
    :return: the anchor image and the list of other image connected
    """

    # find the anchor with most neighbours
    imgs = np.array(img_pairs).ravel()
    imgs, counts = np.unique(imgs, return_counts = True)
    anchor = imgs[np.argmax(counts)]

    # collect neighbours of the anchor
    others = {}
    done = []
    for i, (im1, im2) in enumerate(img_pairs):
        if im1 == anchor:
            others[im2] = LA.inv(H_matrices[i])
            done.append(i)
        elif im2 == anchor:
            others[im1] = H_matrices[i]
            done.append(i)

    # collect the neighbours of the images connected to the anchor
    # stops when there is no update during a iteration
    updates = 1
    while updates != 0:
        updates = 0
        for i, (im1, im2) in enumerate(img_pairs):
            if i in done:
                continue
            if im1 in others:
                others[im2] = np.matmul(others[im1], LA.inv(H_matrices[i]))
                done.append(i)
                updates += 1
            elif im2 in others:
                others[im1] = np.matmul(others[im2], H_matrices[i])
                done.append(i)
                updates += 1

    return anchor, others


def orb(img_names, img_dir):
    """
    Apply orb feature extraction to a list of images
    :param img_names: the names of the images
    :param img_dir: the directory of the images
    :return: the features of each image
    """
    num_features = 1000
    orb = cv2.ORB_create(num_features)
    features = []
    for img in img_names:
        img = os.path.join(img_dir, img)
        img = cv2.imread(img, 0)
        features.append(orb.detectAndCompute(img, None))
    return features


def F_estimation(kp1, kp2):
    """
    Apply Fundamental Matrix Estimation with RANSAC
    :param kp1: key points from the first image
    :param kp2: key points from the second image
    :return: F matrix and mask for inliers
    """
    F, mask = cv2.findFundamentalMat(kp1, kp2, cv2.FM_RANSAC, 3.0)
    F_inlier_count = np.count_nonzero(mask)
    F_inlier_count_ratio = F_inlier_count / len(kp1)
    print("Fundamental matrix estimation")
    print("inliers:", F_inlier_count)
    print("% inliers: {:.2%}".format(F_inlier_count_ratio))
    return F, mask


def H_estimation(kp1, kp2):
    """
    Apply Homography Estimation with RANSAC
    :param kp1: key points from the first image
    :param kp2: key points from the second image
    :return: H matrix and mask for inliers
    """
    H, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC, 3.0)
    H_inlier_count = np.count_nonzero(mask)
    H_inlier_count_ratio = H_inlier_count / len(kp1)
    print("Homography estimation")
    print("inliers:", H_inlier_count)
    print("% inliers: {:.2%}".format(H_inlier_count_ratio))
    return H, mask


def draw_matches(im1, kp1, im2, kp2, matches, mask, out_name):
    """
    Draw matches of keypoint on two images side by side
    and output the image
    :param im1: the first image
    :param kp1: key points in the first image
    :param im2: the second image
    :param kp2: key points in the second image
    :param matches: matches of key points
    :param mask: mask for the matches
    :param out_name: name for the output image
    """
    if mask is not None:
        draw_params = dict(matchesMask = mask.ravel().tolist(), flags = 2)
    else:
        draw_params = dict(flags = 2)
    out_im = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, **draw_params)
    cv2.imwrite(out_name, out_im)


def gaussian_smoothing(im, sigma):
    """
    Apply Gaussian smoothing to image
    :param im: the original image
    :param sigma: sigma for Gaussian
    :return: the smoothed image
    """
    ksize = (int(4 * sigma + 1), int(4 * sigma + 1))
    return cv2.GaussianBlur(im.astype(np.float32), ksize, sigma)


if __name__ == "__main__":
    # Handle the command-line arguments
    if len(sys.argv) != 3:
        print("Usage: %s in_dir out_dir\n" % sys.argv[0])
        sys.exit()

    img_dir = sys.argv[1]
    out_dir = sys.argv[2]

    img_list = os.listdir(img_dir)
    img_list = [name for name in img_list if 'jpg' in name.lower()]
    img_list.sort()

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    features = orb(img_list, img_dir)

    img_pairs = []
    trans_matrices = []

    for i in range(len(img_list)):
        for j in range(i + 1, len(img_list)):
            print('\nProcessing', img_list[i], img_list[j])
            im1_path = os.path.join(img_dir, img_list[i])
            im2_path = os.path.join(img_dir, img_list[j])
            im1_name, im1_type = img_list[i].split('.')
            im2_name, im2_type = img_list[j].split('.')

            im1 = cv2.imread(im1_path, 0)
            im2 = cv2.imread(im2_path, 0)

            kp1, des1 = features[i]
            kp2, des2 = features[j]

            # match two sets of descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x: x.distance)

            # abort if the number of matches is less than 10
            print("Number of matches found:", len(matches))
            if len(matches) < 10:
                print("Too little matches!")
                print("Cannot create mosaic")
                continue

            # draw the matching image
            out_name = im1_name + '_' + im2_name + '_match.' + im1_type
            out_name = os.path.join(out_dir, out_name)
            draw_matches(im1, kp1, im2, kp2, matches, None, out_name)

            # get the matched points
            kpt1_m = np.array([kp1[m.queryIdx].pt for m in matches])
            kpt2_m = np.array([kp2[m.trainIdx].pt for m in matches])

            # Fundamental matrix estimation
            F, F_mask = F_estimation(kpt1_m, kpt2_m)
            out_name = im1_name + '_' + im2_name + '_F.' + im1_type
            out_name = os.path.join(out_dir, out_name)
            draw_matches(im1, kp1, im2, kp2, matches, F_mask, out_name)

            # abort if the percentage matches remained is less than 10
            F_inliers = np.count_nonzero(F_mask)
            if F_inliers / len(matches) < 0.3:
                print("Too little inliers from Fundamental Matrix Estimation!")
                print("Not from the same scene!")
                print("Cannot create mosaic")
                continue

            # Homography estimation
            H, H_mask = H_estimation(kpt1_m, kpt2_m)
            out_name = im1_name + '_' + im2_name + '_H.' + im1_type
            out_name = os.path.join(out_dir, out_name)
            draw_matches(im1, kp1, im2, kp2, matches, H_mask, out_name)

            # abort if the difference between two estimation is greater than 20%
            H_inliers = np.count_nonzero(H_mask)
            differece = (F_inliers - H_inliers) / F_inliers
            print("The % difference of the two estimations {:.2%}".format(differece))
            if H_inliers < 0.8 * F_inliers:
                print("The number of matches from two estimations are too different!")
                print("Not on the same plane!")
                print("Cannot create mosaic")
                continue

            # create mosaic from these two images
            print("Creating mosaic for {} and {}".format(img_list[i], img_list[j]))
            im1 = cv2.imread(im1_path)
            im2 = cv2.imread(im2_path)
            mosaic, _ = create_mosaic(im1, im2, H)
            mosaic = gaussian_smoothing(mosaic, 1)
            out_name = im1_name + '_' + im2_name + '.' + im1_type
            out_name = os.path.join(out_dir, out_name)
            cv2.imwrite(out_name, mosaic)

            # save the transformation for the final mosaic
            img_pairs.append((img_list[i], img_list[j]))
            trans_matrices.append(H)

    # create the mosaic with all the possible images
    if len(img_pairs) != 0:
        anchor, others = find_anchor(img_pairs, trans_matrices)
        mosaic, im_names, im_type = create_full_mosaic(anchor, others, img_dir)
        mosaic = gaussian_smoothing(mosaic, 1)
        im_names.sort()
        print("\nCreating mosaic for", ','.join(im_names))
        out_im_name = '_'.join(im_names) + '.' + im_type
        out_im_name = os.path.join(out_dir, out_im_name)
        cv2.imwrite(out_im_name, mosaic)
