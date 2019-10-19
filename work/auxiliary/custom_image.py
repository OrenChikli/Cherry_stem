import logging
import numpy as np
import cv2
import os
import shutil

from work.auxiliary.exceptions import *
from work.auxiliary import data_functions,decorators
from datetime import datetime

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger_decorator = decorators.Logger_decorator(logger)

COLOR_CODE = {'0': 'blue',
              '1': 'green',
              '2': 'red'}


class CustomImage:

    @logger_decorator.debug_dec
    def __init__(self, img_path=None, mask_path=None,is_binary_mask=False,
                 save_path=None,create_save_dest_flag=True,threshold=None,
                 raw_mask=None, img=None, binary_mask=None):

        self.img_name = os.path.basename(img_path) if img_path is not None else None

        self.img_path = img_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.is_binary_mask=is_binary_mask

        self.img_raw_name = None
        self.extention = None

        self.img = img
        self.raw_mask = raw_mask
        self.binary_mask = binary_mask

        self.threshold = threshold

        self.image_cut = None
        self.green_part = None
        self.brown_part = None

        self.read_data()
        if self.img_name is not None:
            self.get_save_params()

        if create_save_dest_flag:
            self.create_save_dest()

    def create_save_dest(self):

        dir_save_path = data_functions.create_path(self.save_path, self.img_raw_name)
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.save_path = data_functions.create_path(dir_save_path, current_time)


    @logger_decorator.debug_dec
    def move_to(self, dest_path_image, dest_path_label):

        logger.debug(f"moving images to: \n{dest_path_image} and maskes to: \n{dest_path_label}")
        img_path = os.path.join(dest_path_image, self.img_name)
        label_path = os.path.join(dest_path_label, self.img_name)

        _ = shutil.move(self.img_path, img_path)
        _ = shutil.move(self.mask_path, label_path)


    @staticmethod
    def read_img(path,mode=cv2.IMREAD_COLOR):
        try:
            logger.debug(f"Reading image from {path}")
            img = cv2.imread(path, mode | cv2.IMREAD_IGNORE_ORIENTATION)

            if img is None:
                error_message = f"Can't read image from"
                logger.error(error_message, path)
                raise ReadImageException(error_message, path)
            else:
                return img

        except:
            message = f"Failed reading image from: {path}"
            logger.exception(message)
            raise ReadImageException(message)

    @staticmethod
    def read_npy(path):

        logger.info(f"Reading npy file from {path}")
        img = np.load(path)

        if img is None:
            error_message = f"Can't read npy from"
            logger.error(error_message, path)
            raise ReadImageException(error_message, path)
        else:
            return img

    @logger_decorator.debug_dec
    def save_img(self,img, label=''):

        img_name = self.img_raw_name +"." + label + self.extention
        save_path = os.path.join(self.save_path,img_name)
        logger.info(f"saving image: {img_name} to: {self.save_path}")
        cv2.imwrite(save_path, img)

    def get_save_params(self):
        img_raw_name, extention = self.img_name.rsplit(".", 1)
        self.img_raw_name = img_raw_name
        self.extention = '.' + extention

    @logger_decorator.debug_dec
    def read_data(self):

        if self.img_path is not None and self.img is None:
            self.img = self.read_img(self.img_path,cv2.IMREAD_COLOR)
        if self.mask_path is not None:
            if self.is_binary_mask and self.binary_mask is None:
                self.binary_mask = self.read_img(self.mask_path, cv2.IMREAD_GRAYSCALE)
            elif self.raw_mask is None:
                self.raw_mask = self.read_npy(self.mask_path)
                self.binary_mask = self.get_threshold_mask()

    @logger_decorator.debug_dec
    def get_threshold_mask(self):
        threshold_mask = (255 * (self.raw_mask > self.threshold)).astype(np.uint8)
        return threshold_mask

    @logger_decorator.debug_dec
    def get_ontop(self, mask_color=(255,0 ,0),mask=None,
                  display_flag=False,save_flag=False,
                  disp_label='Binary mask ontop of image',save_label='bin_ontop'):

        res = self.img.copy()
        if mask is None:
            mask = self.binary_mask
        mask_inds = (mask == 255)
        res[mask_inds] = mask_color if self.img.shape[2] == 3 else 255
        self.img_save_display(res, disp_label, display_flag, save_flag, save_label)
        return res

    def img_save_display(self,img, disp_label, display_flag, save_flag, save_label):
        if save_flag:
            self.save_img(img, save_label)
        if display_flag:
            self.display_img(img, disp_label)

    @logger_decorator.debug_dec
    def display_img(self,img, disp_label='image'):
        # if img.shape[-1] == 1:
        #     plt.imshow(img[:,:,0])
        # else:
        #     plt.imshow(img)
        cv2.imshow(disp_label,img)

        #cv2.destroyAllWindows()

    @logger_decorator.debug_dec
    def cut_via_mask(self,display_flag=False,save_flag=False,
                  disp_label='Image Cut via mask',save_label='cut'):
        """Cut parts of an image using a mask - get the parts that overlap with the mask"""
        res = cv2.bitwise_and(self.img, self.img, mask=self.binary_mask)
        self.img_save_display(res, disp_label, display_flag, save_flag, save_label)
        return res

    @logger_decorator.debug_dec
    def filter_cut_image(self):
        if self.image_cut is None:
            self.image_cut = self.cut_via_mask()

        hsv = cv2.cvtColor(self.image_cut, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (26, 40, 40), (86, 255, 255))
        mask_brown = cv2.inRange(hsv, (12, 50, 50), (20, 255, 255))
        mask = mask_brown + mask_green
        # mask = cv2.inRange(hsv, (12, 0, 0), (100, 255, 255))
        res = cv2.bitwise_and(self.image_cut, self.image_cut, mask=mask)
        self.img_save_display(res, disp_label, display_flag, save_flag, save_label)
        return res

    @logger_decorator.debug_dec
    def filter_cut_image_green_brown(self):
        if self.image_cut is None:
            self.image_cut = self.cut_via_mask()

        hsv = cv2.cvtColor(self.image_cut, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (26, 40, 40), (86, 255, 255))
        mask_brown = cv2.inRange(hsv, (12, 50, 50), (20, 255, 255))
        mask_sum = np.sum(self.binary_mask)
        pr_green = np.sum(mask_green) / mask_sum
        pr_brown = np.sum(mask_brown) / mask_sum
        self.green_part = cv2.bitwise_and(self.image_cut, self.image_cut, mask=mask_green)
        self.brown_part = cv2.bitwise_and(self.image_cut, self.image_cut, mask=mask_brown)
        return pr_green, pr_brown

    @logger_decorator.debug_dec
    def get_hist_via_mask_cut(self, hist_type='brg', display_flag=False):
        img = self.img.copy()
        if hist_type == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv = cv2.cvtColor(self.cut_via_mask(), cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (26, 40, 40), (86, 255, 255))
        mask_brown = cv2.inRange(hsv, (12, 50, 50), (20, 255, 255))
        mask = mask_brown + mask_green

        hist = []
        for i in range(3):
            curr_hist = cv2.calcHist([img], [i], mask=mask, histSize=[256], ranges=[0, 256])
            hist.append(np.squeeze(curr_hist, axis=1))

        return np.array(hist)

    @logger_decorator.debug_dec
    def get_hist_via_mask(self, hist_type='brg', display_flag=False):
        img = self.img.copy()
        if hist_type == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = self.binary_mask.copy()
        masked_img = self.cut_via_mask()

        hist = []
        for i in range(3):
            curr_hist = cv2.calcHist([img], [i], mask=mask, histSize=[256], ranges=[0, 256])
            hist.append(np.squeeze(curr_hist, axis=1))

        if display_flag:
            fig, ax = plt.subplots(2, 2, figsize=(16, 10))
            ax_flat = ax.flatten()
            ax_flat[0].imshow(img[..., ::-1])
            ax_flat[1].imshow(np.squeeze(mask, axis=2), 'gray')
            ax_flat[2].imshow(masked_img[..., ::-1])
            if hist_type == 'bgr':
                ax_flat[3].plot(hist[0], label='blue', color='blue')
                plt.plot(hist[1], label='green', color='green')
                plt.plot(hist[2], label='red', color='red')
            else:
                ax_flat[3].plot(hist[0], label='hue', color='blue')
                plt.plot(hist[1], label='saturation', color='green')
                plt.plot(hist[2], label='value', color='red')
            plt.xlim([0, 256])
            plt.legend()
            plt.show()

        return np.array(hist)
