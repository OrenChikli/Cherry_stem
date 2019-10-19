import cv2

import numpy as np
import os




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


        #src1_mask = cv2.cvtColor(src1_mask, cv2.COLOR_GRAY2BGR)  # change mask to a 3 channel image


def cut_via_color(img_path,dest_path,thres=50):
    """ method to create maskes from stem classes- stems on black background.
    uses simple thresholding on images loaded as grayscale"""
    for img_entry in os.scandir(img_path):
        img = cv2.imread(img_entry.path,cv2.IMREAD_GRAYSCALE)
        res = 255 * (img > thres).astype(np.uint8)
        cv2.imwrite(os.path.join(dest_path,img_entry.name),res)



