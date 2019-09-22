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

logger = logging.getLogger(__name__)



class Image:

    def __init__(self, img_path, mask_path=None):

        self.image_name = os.path.basename(img_path)
        self.img_path = img_path
        self.img = None

        self.mask_path = mask_path
        self.mask = None
        self.cut_mask=None
        self.mask_mean_h = None

        self.segmentation = None

        self.read_local()

    def move_to(self, dest_path_image, dest_path_label):
        logger.debug("-> move_to")
        logger.debug(f"moving images to {dest_path_image} and maskes to {dest_path_label}")
        img_path = os.path.join(dest_path_image, self.image_name)
        label_path = os.path.join(dest_path_label, self.image_name)

        _ = shutil.move(self.img_path, img_path)
        _ = shutil.move(self.mask_path, label_path)

        logger.debug(" <- move_to")

    def read_local(self):

        logger.debug(" -> read")
        logger.debug("Reading image %s", self.img_path)

        try:
            logger.debug("Reading image locally by OpenCV")
            self.img = cv2.imread(self.img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        except:
            logger.exception('Failed reading image: ' + self.img_path)
            raise ReadImageException('Failed reading image: ' + self.img_path)

        if self.mask_path:
            logger.debug("Reading mask %s", self.mask_path)

            try:
                logger.debug("Reading mask image locally by OpenCV")
                self.mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)

            except:
                logger.exception('Failed reading  mask image: ' + self.mask_path)
                raise ReadImageException('Failed reading  mask image: ' + self.mask_path)

        if self.img is None:
            error_message = "Can't read image from "
            logger.error(error_message, self.img_path)
            raise ReadImageException(error_message, self.img_path)

        if self.mask_path is not None and self.mask is None:
            error_message = "Can't read mask from "
            logger.error(error_message, self.mask_path)
            raise ReadImageException(error_message, self.mask_path)

        logger.debug(" <- read")

    def cut_via_mask(self,save_flag=False,dest_path=None):
        """Cut parts of an image using a mask - get the parts that overlap with the mask"""
        out = cv2.subtract(self.mask, self.img)
        self.cut_mask = cv2.subtract(self.mask, out)
        if save_flag:
            cv2.imwrite(os.path.join(dest_path, self.image_name), out)

    def get_mean_hue(self,save_flag=False,dest_path=None):
        cut_mask_not_zero = self.cut_mask[self.cut_mask > 0]
        cut_mask_hsv = cv2.cvtColor(self.cut_mask, cv2.COLOR_RGB2HSV) * cut_mask_not_zero
        cut_mask_h = cut_mask_hsv[:, :, 0]
        cut_mask_h = cut_mask_h[cut_mask_h > 0]
        cut_mask_h_rad = np.deg2rad(cut_mask_h)
        mean_hue = np.rad2deg(circmean(cut_mask_h_rad))
        hsv = ((mean_hue, 255, 255) * np.ones_like(self.img)).astype(np.uint8)
        res_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        self.mask_mean_h = res_img
        if save_flag:
            cv2.imwrite(os.path.join(dest_path, self.image_name), res_img)