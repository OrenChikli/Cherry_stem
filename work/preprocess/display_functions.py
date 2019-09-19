import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay(img,mask,alpha=1):
        out = cv2.addWeighted(mask, alpha, img, 1 - alpha, 0)
        return out


def put_binary_ontop(img,mask,mask_color=(255, 255, 0),alpha = 1):
    '''
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
    alpha: float [0, 1].

    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = out.copy()

    if img.shape[-1] != 3:
        mask_color = 255
        out[np.where(mask)] = mask_color
    else:
        bin_bool = np.where(mask)
        img_layer[np.where(mask)] = mask_color
        out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    return out


def canny(img):

    img = cv2.imread('messi5.jpg', 0)
    edges = cv2.Canny(img, 100, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()