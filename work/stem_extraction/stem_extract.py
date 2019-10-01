from auxiliary import data_functions
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

import logging
from auxiliary.exceptions import *


logger = logging.getLogger(__name__)

from auxiliary.custom_image import CustomImage

OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Chi-Squared", cv2.HISTCMP_CHISQR),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))


HIST_TYPE = {'rgb': lambda x: x.get_hist_via_mask,
             'hsv': lambda x: x.get_hsv_hist}

class StemExtractor:

    def __init__(self, img_path, mask_path, src_path, threshold=0.5):

        self.img_path = img_path
        self.mask_path = mask_path
        self.threshold = threshold

        self.thres_save_path = data_functions.create_path(src_path, f"thres_{threshold}")

        self.threshold_masks_path = None

        self.cut_image_path = None
        self.ontop_path=None


        self.groud_truth_hist_dict = None



    @staticmethod
    def get_clases_dict(classes_path):
        classes_dict = {}
        for img_entry in os.scandir(classes_path):
            img_color = cv2.imread(img_entry.path, cv2.IMREAD_COLOR)
            img_color = img_color[0, 0]
            class_name = img_entry.name.split('.')[0]
            classes_dict[class_name] = img_color
        return classes_dict





    def get_threshold_masks(self):
        """
        return a thresholder version of the input grayscale mask where the mask is positive for points that are greater
        than the given threshold
        :return:
        """
        self.threshold_masks_path = data_functions.create_path(self.thres_save_path, f'binary')
        for img in tqdm(self.image_obj_iterator()):
            img_save_path = os.path.join(self.threshold_masks_path,img.image_name)
            cv2.imwrite(img_save_path, img.threshold_mask)


    def get_stems(self):
        """
        extract the "stems" from the image - return the areas in the image that are activated in the thresholded mask
        eg if pixel (156,46) is turned on in the mask image, it will show in the result
        :return:
        """
        logger.debug(" <- get_stems")
        self.cut_image_path = data_functions.create_path(self.thres_save_path, f'stems')
        logger.info(f"creting stems in {self.cut_image_path}")
        for img in tqdm(self.image_obj_iterator()):
            img.cut_via_mask()
            cv2.imwrite(os.path.join(self.cut_image_path, img.image_name), img.image_cut)

        logger.debug(" -> get_stems")

    def get_ontop_images(self):
        """
        return the overlay of the thresholded mask on top of the source image
        :return:
        """
        logger.debug(" <- get_notop")
        self.ontop_path = data_functions.create_path(self.thres_save_path, f'on_top')
        logger.info(f"creting ontop images in {self.ontop_path}")
        for img in tqdm(self.image_obj_iterator()):
            img.get_ontop()
            cv2.imwrite(os.path.join(self.ontop_path, img.image_name), img.ontop)

        logger.debug(" -> get_notop")


    def fillter_via_color(self,lower,upper):
        logger.debug(" <- fillter_via_color")
        out_path = data_functions.create_path(self.thres_save_path, f'filtered')
        logger.info("saving lower and upper values")
        save_dict = dict(lower=lower,upper=upper)
        data_functions.save_json(save_dict,"fillter_range.json",out_path)
        logger.info(f"creting filltered images in {out_path}")
        for img in tqdm(self.image_obj_iterator()):
            img.filter_cut_iamge(lower=lower,upper=upper)
            cv2.imwrite(os.path.join(out_path, img.image_name), img.filtered_cut)

        logger.debug(" -> get_notop")



    def image_obj_iterator(self):
        for img_entry in os.scandir(self.img_path):
            mask_name = img_entry.name.rsplit(".",1)[0]+'.npy'
            mask_path = os.path.join(self.mask_path, mask_name)
            img = CustomImage(img_entry.path, mask_path,threshold=self.threshold)
            yield img


    def calc_hists(self,hist_type='brg'):
        dest_path = data_functions.create_path(self.thres_save_path, f'histograms')

        for img in tqdm(self.image_obj_iterator()):
            img_raw_name = img.image_name.split('.')[0]
            curr_dest_path = os.path.join(dest_path,f"{img_raw_name}.npy")
            fig_big_hist= img.get_hist_via_mask(hist_type=hist_type)
            np.save(curr_dest_path,fig_big_hist)















