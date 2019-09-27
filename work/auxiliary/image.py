import logging
import numpy as np
import cv2
import os
import shutil
from work.auxiliary.exceptions import ReadImageException

from scipy.stats import circmean
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

COLOR_CODE = {'0':'blue',
              '1':'green',
              '2':'red'}


class Image:

    def __init__(self, img_path, mask_path=None,threshold=None):

        self.image_name = os.path.basename(img_path)
        self.img_path = img_path
        self.img = None

        self.mask_path = mask_path
        self.grayscale_mask = None

        self.threshold = threshold
        self.threshold_mask=None


        self.image_cut = None

        self.mask_mean_h = None
        self.mask_mean_color=None
        self.segmentation = None

        self.read_local()
        if threshold is not None:
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

        logger.debug(" <- read")

    def get_threshold_mask(self):
        logger.debug(" <- get_threshold_mask")
        self.threshold_mask = (255*(self.grayscale_mask > self.threshold)).astype(np.uint8)
        logger.debug(" -> get_threshold_mask")

    def get_sharp_mask(self):

        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

        self.grayscale_mask = cv2.filter2D(self.grayscale_mask, -1, kernel)

        if self.threshold_mask is not None:
            self.get_threshold_mask()





    def cut_via_mask(self):
        """Cut parts of an image using a mask - get the parts that overlap with the mask"""
        if self.threshold_mask is None:
            self.get_threshold_mask()

        color_mask = cv2.cvtColor(self.threshold_mask, cv2.COLOR_GRAY2RGB)
        out = cv2.subtract(color_mask, self.img)
        self.image_cut = cv2.subtract(color_mask, out)


    def get_mean_hue(self):
        if self.image_cut == None:
            self.cut_via_mask()

        hsv = cv2.cvtColor(self.image_cut, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        h_of_non_zero_areas = h[self.threshold_mask > 0]
        h_rad = np.deg2rad(h_of_non_zero_areas)
        mean_h = np.rad2deg(circmean(h_rad))
        res_hsv = ((mean_h, 255, 255) * np.ones((50,50,3))).astype(np.uint8)
        res_img = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2RGB)
        self.mask_mean_h = res_img



    def get_mean_color(self):
        out = cv2.mean(self.img, self.threshold_mask)[:-1]
        res_img = np.ones(shape=(50,50,3), dtype=np.uint8) * np.uint8(out)
        self.mask_mean_color=res_img

    def get_hist_via_mask(self,return_hist=True,display_flag=False):
        img= self.img
        mask = self.threshold_mask
        self.cut_via_mask()
        masked_img = self.image_cut

        hist_blue = cv2.calcHist([img], [0], mask, [256], [0, 256])
        hist_green = cv2.calcHist([img], [1], mask, [256], [0, 256])
        hist_red = cv2.calcHist([img], [2], mask, [256], [0, 256])


        if display_flag:
            fig, ax = plt.subplots(2, 2, figsize=(16, 10))
            ax_flat = ax.flatten()
            ax_flat[0].imshow(img[...,::-1])
            ax_flat[1].imshow(mask, 'gray')
            ax_flat[2].imshow(masked_img[...,::-1])
            ax_flat[3].plot(hist_blue, label='blue', color='blue')
            plt.plot(hist_green, label='green', color='green')
            plt.plot(hist_red, label='red', color='red')
            plt.xlim([0, 256])
            plt.legend()
            plt.show()

        if return_hist:
            ret_hist = np.hstack((hist_blue, hist_green, hist_red))
            individual_hists = {'blue':hist_blue,'green':hist_green,'red':hist_red}
            return ret_hist, individual_hists

    """
    def get_hist_via_mask(self,return_hist=True,display_flag=False):
        img= self.img
        mask = self.threshold_mask
        self.cut_via_mask()
        masked_img = self.image_cut

        hist_blue = cv2.calcHist([img], [0], mask, [256], [0, 256])
        hist_green = cv2.calcHist([img], [1], mask, [256], [0, 256])
        hist_red = cv2.calcHist([img], [2], mask, [256], [0, 256])

        if display_flag:
            fig, ax = plt.subplots(2, 2, figsize=(16, 10))
            ax_flat = ax.flatten()
            ax_flat[0].imshow(img[...,::-1])
            ax_flat[1].imshow(mask, 'gray')
            ax_flat[2].imshow(masked_img[...,::-1])
            ax_flat[3].plot(hist_blue, label='blue', color='blue')
            plt.plot(hist_green, label='green', color='green')
            plt.plot(hist_red, label='red', color='red')
            plt.xlim([0, 256])
            plt.legend()
            plt.show()

        if return_hist:
            ret_hist = cv2.calcHist([img], [0, 1, 2], mask, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            ret_hist = cv2.normalize(ret_hist, ret_hist).flatten()
            individual_hists = {'blue':hist_blue,'green':hist_green,'red':hist_red}
            return ret_hist, individual_hists
    """

"""def get_hist_via_mask(self, return_hist=False):
    img = self.img
    mask = self.threshold_mask
    self.cut_via_mask()
    masked_img = self.image_cut

    hist_blue = cv2.calcHist([img], [0], mask, [256], [0, 256])
    hist_green = cv2.calcHist([img], [1], mask, [256], [0, 256])
    hist_red = cv2.calcHist([img], [2], mask, [256], [0, 256])

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    ax_flat = ax.flatten()
    ax_flat[0].imshow(img[..., ::-1])
    ax_flat[1].imshow(mask, 'gray')
    ax_flat[2].imshow(masked_img[..., ::-1])
    ax_flat[3].plot(hist_blue, label='blue', color='blue')
    plt.plot(hist_green, label='green', color='green')
    plt.plot(hist_red, label='red', color='red')
    plt.xlim([0, 256])
    plt.legend()
    plt.show()
    if return_hist:
        return {'blue': hist_blue, 'green': hist_green, 'red': hist_red}"""



"""
        masked_img = self.image_cut
        fig,ax = plt.figure(fig_size=(14,8))
        hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
        plt.subplot(221), plt.imshow(img)
        plt.subplot(222), plt.imshow(mask, 'gray')
        plt.subplot(223), plt.imshow(masked_img)
        plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
        plt.xlim([0, 256])
        plt.show()"""
