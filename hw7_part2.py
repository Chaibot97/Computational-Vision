"""
Author:   Lizhou Cai
Course:   CSCI 4270
File: hw7_part2.py

Purpose: Do optical flow on pairs of images
    to detect motion of the camera and objects

"""

import cv2
import numpy as np
import numpy.linalg as LA

import os
import sys


def is_inlier(p1, p2, foe, tau):
    """
    check if a line through two points is an inlier to foe with tolerance of tau
    :param p1: the first point of the line
    :param p2: the second point of the line
    :param foe: the focus of expansion
    :param tau: the tolerance
    :return: if the line is a inlier
    """
    p1 = np.concatenate((p1, [1]))
    p2 = np.concatenate((p2, [1]))
    a, b, c = np.cross(p1, p2)
    return np.abs(np.dot(foe, (a, b)) + c) / LA.norm([a, b]) < tau


def get_inliers(kp1, kp2, foe, tau):
    """
    return the inliers and outliers indices of motion vectors
    :param kp1: set of points before motion
    :param kp2: set of points after motion
    :param foe: the focus of expansion
    :param tau: the tolerance
    :return: inliers and outliers indices
    """
    inliers = []
    outliers = []
    for i in range(len(kp1)):
        if is_inlier(kp1[i], kp2[i], foe, tau):
            inliers.append(i)
        else:
            outliers.append(i)
    return np.array(inliers), np.array(outliers)


def get_intersect(pair1, pair2):
    """
    Returns the point of intersection
    of two lines passing through two pairs of points
    :param pair1: first pair of points
    :param pair2: second pair of points
    :return: the intersection
    """
    # calculate the homogeneous coords
    tmp = np.vstack((pair1, pair2))
    h = np.hstack((tmp, np.ones((4, 1))))

    # line through each pair of points
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])

    # get the intersect
    x, y, z = np.cross(l1, l2)
    x /= z
    y /= z
    return x, y


def RANSAC(kp1, kp2, iterations):
    """
    Do RANSAC to find the best foe
    :param kp1: keypoints before the motion
    :param kp2: keypoints after the motion
    :param iterations: the number of iterations for RANSAC
    :return: best feo and corresponding inliers and outliers
    """
    k_max = 0
    m_foe = (0, 0)
    m_inliers = []
    m_outliers = []
    for k in range(iterations):
        # random select 2 different points as sample
        sample = np.random.randint(0, len(kp1), 2)
        if sample[0] == sample[1]:
            continue

        # calculate the line through the 2 points
        p1 = kp1[sample[0]], kp2[sample[0]]
        p2 = kp1[sample[1]], kp2[sample[1]]

        # the intersection
        foe = get_intersect(p1, p2)
        if foe == (np.inf, np.inf):
            continue

        # calculate the inliers and outliers
        inliers, outliers = get_inliers(kp1, kp2, foe, 5)

        # update the best feo
        if len(inliers) > k_max:
            k_max = len(inliers)
            m_foe = foe
            m_inliers = inliers
            m_outliers = outliers

    return k_max, m_foe, m_inliers, m_outliers


def estimate_foe(start, end):
    """
    Do least-squares estimate to estimate the foe
    :param start: starting points
    :param end:ending points
    :return: the estimated foe
    """
    A = np.zeros(start.shape)
    diff = end - start
    A[:, 0] = diff[:, 1]
    A[:, 1] = -diff[:, 0]
    b = start[:, 0] * diff[:, 1] - start[:, 1] * diff[:, 0]
    foe = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    foe = (int(foe[0]),int(foe[1]))
    return foe


def draw_circles(img, points, color):
    """
    draw circles on a image
    :param img: the original image
    :param points: the centers of circles to be drawn
    :param color: the color of the circles
    :return: the image with circles
    """
    for p in points:
        img = cv2.circle(img, (p[0], p[1]), 5, color, thickness=2)
    return img


def draw_arrows(img, p1, p2, color):
    """
    draw arrow line on a image
    :param img: the original image
    :param p1: the starting points of the arrows
    :param p2: the ending points of the arrows
    :param color: the color of arrows
    :return: the image with arrows
    """
    for i in range(p1.shape[0]):
        x = tuple(p1[i].ravel())
        y = tuple(p2[i].ravel())
        img = cv2.arrowedLine(img, x, y, color, thickness=3)
    return img


def draw_clusters(img, p1, p2, k, label, thres, padding):
    """
    draw clusters points and motion vectors with bounding boxes
    :param img: the original image
    :param p1: the set of starting points
    :param p2: the set of ending points
    :param k: number of clusters
    :param label: label for each data point
    :param thres: threshold to get rid of small clusters
    :param padding: the padding for the bounding box
    :return: the result image
    """
    for i in range(k):
        color = np.random.uniform(low=0, high=255, size=3)
        index = np.where(label == i)[0]
        if len(index) <= thres:
            continue

        # plot for one cluster
        start = p1[index]
        end = p2[index]
        img = draw_circles(img, start, color)
        img = draw_circles(img, end, color)
        img = draw_arrows(img, start, end, color)
        min_x, min_y = np.amin(end, axis=0).astype(int) - padding
        max_x, max_y = np.amax(end, axis=0).astype(int) + padding
        img = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color, 2)
    return img


if __name__ == "__main__":
    # Handle the command-line arguments
    if len(sys.argv) != 2:
        print("Usage: %s in_path\n" % sys.argv[0])
        sys.exit()

    in_path = sys.argv[1]

    if not os.path.exists('result/'):
        os.mkdir('result/')

    img_list = os.listdir(in_path)
    img_list = [name for name in img_list
                if 'jpg' in name.lower()
                or 'png' in name.lower()]
    img_list.sort()

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=200,
                          qualityLevel=0.5,
                          minDistance=10,
                          blockSize=3)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(10, 10),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # read in 2 images at a time
    for i in range(0, len(img_list), 2):
        j = i + 1
        print('\nProcessing', img_list[i], img_list[j])
        im1_path = os.path.join(in_path, img_list[i])
        im2_path = os.path.join(in_path, img_list[j])
        im1_name, im1_type = img_list[i].split('.')
        im2_name, im2_type = img_list[j].split('.')
        name = im1_name.split('_')[0]

        im1 = cv2.imread(im1_path)
        im2 = cv2.imread(im2_path)
        im1_gry = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gry = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        kp1 = cv2.goodFeaturesToTrack(im1_gry, mask=None, **feature_params)

        # calculate optical flow
        kp2, st, err = cv2.calcOpticalFlowPyrLK(im1_gry, im2_gry, kp1, None, **lk_params)

        # Select good points
        kp1 = kp1[st == 1]
        kp2 = kp2[st == 1]

        # RANSAC
        k_max, foe, inliers, outliers = RANSAC(kp1, kp2, 50)
        foe = (int(foe[0]), int(foe[1]))
        print('Number of keypoints %d' % len(kp1))
        print('foe from RANSAC', foe)
        print('Number of inliers %d' % k_max)
        rate = k_max/len(kp1)
        print("percent of inliers {:.1%}".format(rate))

        inliers1, inliers2 = kp1[inliers], kp2[inliers]
        outliers1, outliers2 = kp1[outliers], kp2[outliers]

        im_out = im2.copy()
        if rate < 0.15:
            print("Not enough inliers, camera is not moving")
        else:
            foe_est = estimate_foe(inliers1, inliers2)
            print("least-squares estimate of feo:", foe_est)
            im_out = cv2.circle(im_out, (int(foe[0]), int(foe[1])), 15, (0, 0, 255), thickness=-1)

        # draw the motion vectors and keypoints
        im_out = draw_circles(im_out, kp1, (0, 255, 255))
        im_out = draw_circles(im_out, kp2, (0, 255, 0))
        im_out = draw_arrows(im_out, inliers1, inliers2, (0, 0, 255))
        im_out = draw_arrows(im_out, outliers1, outliers2, (255, 0, 0))

        out_name = 'result/' + name + '_output1.jpg'
        cv2.imwrite(out_name, im_out)

        # do kmeans clustering on the outlires left
        all_values = outliers2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        num_reinitializations = 30
        k = min(5, len(outliers2))
        initialization_method = cv2.KMEANS_PP_CENTERS
        ret, label, center = cv2.kmeans(all_values, k, None, criteria,
                                        num_reinitializations, initialization_method)

        # draw the clusters
        im_out2 = im2.copy()
        out_name = 'result/' + name + '_output2.jpg'
        center = np.uint8(center)

        # get rid of small clusters
        thres = 0.2 * len(outliers2)
        im_out2 = draw_clusters(im_out2, outliers1, outliers2, k, label, thres, 20)

        cv2.imwrite(out_name, im_out2)
