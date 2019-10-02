import cv2
import numpy as np


def get_mask(img, color, diff_range=20):
    lower = color - diff_range
    upper = color + diff_range
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)
    return output, mask


def color():
    img = cv2.imread(
        r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46\thres_0.4\stems\78856-09861.png.jpg')

    brown = np.array([145, 80, 40], dtype=np.uint8)  # RGB
    green = np.array([127, 240, 0], dtype=np.uint8)
    diff = 20

    # get_pr_color(brown,'brown', diff, img)
    get_pr_color(green, 'green', diff, img)


def get_pr_color(color, color_name, diff, img):
    output, mask = get_mask(img, color, diff)
    ratio_brown = cv2.countNonZero(mask) / (img.size / 3)
    print(f'{color_name} pixel percentage:', np.round(ratio_brown * 100, 2))
    cv2.imshow("images", np.hstack([img, output]))
    cv2.waitKey(0)


def bgr_to_hex(bgr):
    rgb = list(bgr)
    rgb.reverse()
    return tuple(rgb)


def FindColors(image):
    color_hex = []
    for i in image:
        for j in i:
            j = list(j)
            color_hex.append(bgr_to_hex(tuple(j)))
    return set(color_hex)


def func():
    img = cv2.imread(
        r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46\thres_0.4\stems\78856-09861.png.jpg')

    color_list = FindColors(img)
    print(color_list)


import binascii
import struct
from PIL import Image

import scipy
import scipy.misc
import scipy.cluster


def func2():
    NUM_CLUSTERS = 6

    im = Image.open(
        r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46\thres_0.4\stems\78856-09861.png.jpg')
    # im = im.resize((150, 150))      # optional, to reduce time
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))
    shape = (150, 150, 1)
    im1 = cv2.imread(
        r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46\thres_0.4\stems\38357-02789.png.jpg')
    cv2.imshow('image', im1)
    for i in range(len(codes)):
        peak = np.ones(shape) * codes[i].reshape(1, -1)
        peak = peak.astype(np.uint8)
        cv2.imshow(f"{i}", peak)
        print(codes[i].astype(np.uint8))
    cv2.waitKey(0)
    # count occurrences
    print(bins)