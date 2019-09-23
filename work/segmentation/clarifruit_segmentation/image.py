import logging
import numpy as np
import cv2
import os
import shutil
from .exceptions import ReadImageException
from .utils import Utils
from .segmentation import Segmentation
from skimage.util import img_as_float
from scipy.stats import circmean
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)



class Image:

    def __init__(self, img_path, mask_path=None, cut_mask_path=None,threshold=0.5):

        self.image_name = os.path.basename(img_path)
        self.img_path = img_path
        self.img = None

        self.mask_path = mask_path
        self.grayscale_mask = None
        self.sharp_mask = None
        self.threshold = threshold



        self.cut_mask_path=cut_mask_path
        self.cut_mask = None

        self.mask_mean_h = None
        self.mask_mean_color=None
        self.segmentation = None

        self.read_local()
        self.get_threshold_mask()

    def move_to(self, dest_path_image, dest_path_label):
        logger.debug("-> move_to")
        logger.debug(f"moving images to {dest_path_image} and maskes to {dest_path_label}")
        img_path = os.path.join(dest_path_image, self.image_name)
        label_path = os.path.join(dest_path_label, self.image_name)

        _ = shutil.move(self.img_path, img_path)
        _ = shutil.move(self.mask_path, label_path)

        logger.debug(" <- move_to")

    @staticmethod
    def read_img(path, img_type):
            logger.debug(f"Reading {img_type} {path}")

            try:
                logger.debug("Reading mask image locally by OpenCV")
                if img_type == 'mask':
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
                else:
                    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if img is None:
                    error_message = f"Can't read {img_type} from"
                    logger.error(error_message, path)
                    raise ReadImageException(error_message, path)
                else:
                    return img

            except:
                message = f"Failed reading {img_type}: {path}"
                logger.exception(message)
                raise ReadImageException(message)


    def read_local(self):

        logger.debug(" -> read")

        self.img = self.read_img(self.img_path,'image')
        if self.mask_path is not None:
            self.grayscale_mask = self.read_img(self.mask_path, 'mask')
        if self.cut_mask_path is not None:
            self.cut_mask = self.read_img(self.cut_mask_path,'cut mask')

        logger.debug(" <- read")

    def get_threshold_mask(self):
        x= 255*(self.grayscale_mask > 255*self.threshold)
        self.threshold_mask = (255*(self.grayscale_mask > 255*self.threshold)).astype(np.uint8)


    def cut_via_mask(self,save_flag=False,dest_path=None):
        """Cut parts of an image using a mask - get the parts that overlap with the mask"""
        color_mask = cv2.cvtColor(self.threshold_mask, cv2.COLOR_GRAY2RGB)

        out = cv2.subtract(color_mask, self.img)
        self.cut_mask = cv2.subtract(color_mask, out)
        if save_flag:
            cv2.imwrite(os.path.join(dest_path, self.image_name), self.cut_mask)



    def get_mean_hue(self, save_flag=False, dest_path=None):
        color_mask = cv2.cvtColor(self.grayscale_mask, cv2.COLOR_GRAY2RGB) / 255
        weighted = (self.img * color_mask).astype(np.uint8)
        weighted_hsv = cv2.cvtColor(weighted, cv2.COLOR_RGB2HSV)
        weighted_h = weighted_hsv[:, :, 0]
        weighted_h_thres = weighted_h[weighted_h > 0]
        h_rad = np.deg2rad(weighted_h_thres)
        mean_h = np.rad2deg(circmean(h_rad))
        hsv = ((mean_h, 255, 255) * np.ones_like(self.img)).astype(np.uint8)
        res_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        self.mask_mean_h = res_img
        if save_flag:
            cv2.imwrite(os.path.join(dest_path, self.image_name), res_img)

    def sharpen_mask(self,save_flag=False,dest_path=None):

        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        self.sharp_mask = cv2.filter2D(self.grayscale_mask, -1, kernel)
        if save_flag:
            cv2.imwrite(os.path.join(dest_path, self.image_name), self.sharp_mask)

        # smoothed = cv2.GaussianBlur(self.mask,(5,5),0)
        # cv2.imshow("smoothed",smoothed)
        # res = cv2.subtract(self.mask,smoothed)



    # def get_mean_hue(self,save_flag=False,dest_path=None):
    #     cut_mask_not_zero = self.cut_mask > 0
    #     cut_mask_hsv = cv2.cvtColor(self.cut_mask, cv2.COLOR_RGB2HSV) * cut_mask_not_zero
    #     cut_mask_h = cut_mask_hsv[:, :, 0]
    #     cut_mask_h = cut_mask_h[cut_mask_h > 0]
    #     cut_mask_h_rad = np.deg2rad(cut_mask_h)
    #     mean_hue = np.rad2deg(circmean(cut_mask_h_rad))
    #     hsv = ((mean_hue, 255, 255) * np.ones_like(self.img)).astype(np.uint8)
    #     res_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    #     self.mask_mean_h = res_img
    #     if save_flag:
    #         cv2.imwrite(os.path.join(dest_path, self.image_name), res_img)

    def get_mean_color(self,save_flag=False,dest_path=None):
        out = cv2.mean(self.img, self.grayscale_mask)[:-1]
        res_img = np.ones(shape=self.img.shape, dtype=np.uint8) * np.uint8(out)
        self.mask_mean_color=res_img
        if save_flag:
            cv2.imwrite(os.path.join(dest_path, self.image_name), res_img)
