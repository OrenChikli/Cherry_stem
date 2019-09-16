import cv2
import skimage.transform as trans
import os
import numpy as np



def combine():
    img_name = '74714-32897.png.jpg'
    img_path = r'D:\OneDrive - Open University of Israel\clarifruit_labeling\cherry\image'
    mask_path = r'D:\OneDrive - Open University of Israel\clarifruit_labeling\cherry\label'

    curr_img_path = os.path.join(img_path,img_name)
    curr_mask_path = os.path.join(mask_path, img_name)
    img = cv2.imread(curr_img_path,cv2.IMREAD_COLOR)
    mask = cv2.imread(curr_mask_path, cv2.IMREAD_GRAYSCALE)

    mask = trans.resize(mask, img.shape)
    mask = (255 * mask).astype(np.uint8)
    mask = img - mask
    alpha = 1

    img_mask = cv2.addWeighted(mask, alpha, img, 1 - alpha, 0)
    cv2.imshow("with mask",img_mask)
    cv2.waitKey(0)


if __name__ == '__main__':
    combine()