import logging
import numpy as np
import cv2
import os
import shutil

from work.auxiliary.exceptions import *
from work.auxiliary import data_functions,decorators
from datetime import datetime
import pickle

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
            img_raw_name, extention = self.img_name.rsplit(".", 1)
            self.img_raw_name = img_raw_name
            self.extention = '.' + extention

        if create_save_dest_flag:
            self.create_save_dest()

    @logger_decorator.debug_dec
    def create_save_dest(self):
        """
        Create a folder with the current time stamp the destination folder,
        where any results to be save to file will be saved to
        :return:
        """
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
        """
        A method to read an image using cv.imread with exceptions
        :param path: source to the image
        :param mode: one of the cv2 imread enumerations specified in
        https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
        :return: cv2.imread result if not None
        """
        try:
            logger.debug(f"Reading image from {path}")
            img = cv2.imread(path, mode | cv2.IMREAD_IGNORE_ORIENTATION)

            if img is None:
                error_message = f"Can't read image from"
                raise ReadImageException(error_message, path)
            else:
                return img

        except:
            message = 'Problem reading image file'
            logger.exception(message)


    @staticmethod
    def read_npy(path):
        """
        a method to read npy files
        :param path: the path to the file
        :return:
        """
        try:
            logger.debug(f"Reading npy file from {path}")
            img = np.load(path)

            if img is None:
                raise ReadImageException()
            else:
                return img
        except:
            error_message = f"Can't read npy from"
            logger.error(error_message, path)
            raise ReadImageException(error_message,path)


    @logger_decorator.debug_dec
    def save_img(self,img, label=''):
        """
        A method to save a given image to self.save_path
        :param img: np.array image to be saved
        :param label: optional, str, an added label to the file name for later
        identification
        :return:
        """

        img_name = self.img_raw_name +"." + label + self.extention
        save_path = os.path.join(self.save_path,img_name)
        logger.debug(f"saving image: {img_name} to: {self.save_path}")
        cv2.imwrite(save_path, img)




    @logger_decorator.debug_dec
    def read_data(self):
        """
        a method to read the data given in the "self.img_path", "self.mask_path"
        if these are not None
        :return:
        """

        if self.img_path is not None and self.img is None:
            self.img = self.read_img(self.img_path,cv2.IMREAD_COLOR)
        if self.mask_path is not None:
            if self.is_binary_mask and self.binary_mask is None:
                self.binary_mask = self.read_img(self.mask_path,
                                                 cv2.IMREAD_GRAYSCALE)
            elif self.raw_mask is None:
                self.raw_mask = self.read_npy(self.mask_path)

        if self.binary_mask is not None:
            # some sources are not exactly binary
            self.binary_mask = (255 * (self.binary_mask > 150)).astype(np.uint8)

        if self.binary_mask is None and self.raw_mask is not None:
            self.binary_mask = self.get_threshold_mask()


    @logger_decorator.debug_dec
    def get_threshold_mask(self):
        """
        return a binary mask using the "self.threshold" and "self.raw_mask"
        where a pixel has a value of 255 if it's value in self.raw_mask is
        greater than self.threshold
        :return:
        """
        return (255 * (self.raw_mask > self.threshold)).astype(np.uint8)


    @logger_decorator.debug_dec
    def get_ontop(self, mask_color=(255,0 ,0),mask=None,
                  display_flag=False,save_flag=False,
                  disp_label='Binary mask ontop of image',save_label='bin_ontop'):
        """
        Display the a binary mask on top of the image
        :param mask_color: tuple, the color of the mask on the result,
        default (255,0,0)
        :param mask: optional, np.array, a mask image to be used, if None
        self.binary_mask
        is used
        :param display_flag: optional,bool, whether to display the result
        :param save_flag: optional,bool, whether to save the result to file
        :param disp_label: optional,str, the display label if display_flag is
        True
        :param save_label: optional,str, a label which is added to the file
        name of the image to save if display_flag is True
        :return: the image with the mask ontop
        """

        res = self.img.copy()
        if mask is None:
            mask = self.binary_mask
        mask_inds = (mask == 255)
        res[mask_inds] = mask_color if self.img.shape[2] == 3 else 255
        if save_flag:
            self.save_img(res, save_label)
        if display_flag:
            self.display_img(res, disp_label)
        return res




    @logger_decorator.debug_dec
    def display_img(self,img, disp_label='image'):
        """
        a function to display images, exist's for easy modification in the
        future
        :param img: np.array, the image to be displayed
        :param disp_label: str, the title of the image
        :return:
        """

        cv2.imshow(disp_label,img)



    @logger_decorator.debug_dec
    def cut_via_mask(self,display_flag=False,save_flag=False,
                  disp_label='Image Cut via mask',save_label='cut'):
        """
        Cut parts of an image using a mask - get the parts that overlap with
        the mask and save to a new image
        :param display_flag: optional,bool, whether to display the result
        :param save_flag: optional,bool, whether to save the result to file
        :param disp_label: optional,str, the display label if display_flag is
        True
        :param save_label: optional,str, a label which is added to the file
        name of the image to save if display_flag is True
        :return: parts of the source image that overlaps with the mask
        """
        res = cv2.bitwise_and(self.img, self.img, mask=self.binary_mask)
        if save_flag:
            self.save_img(res, save_label)
        if display_flag:
            self.display_img(res, disp_label)
        return res


