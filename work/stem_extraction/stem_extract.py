from auxiliary import data_functions
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

import logging
from auxiliary.exceptions import *
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
import pandas as pd

PR_DICT ={0.9:'A',0.6:'B',0.3:'C',0:'D'}


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

    def __init__(self, img_path, mask_path, save_path, threshold=0.5, use_thres_flag=True):

        self.img_path = img_path
        self.mask_path = mask_path
        self.threshold = threshold
        if use_thres_flag:
            self.thres_save_path = data_functions.create_path(save_path, f"thres_{threshold}")
        else:
            self.thres_save_path = save_path

        self.threshold_masks_path = None

        self.cut_image_path = None
        self.ontop_path=None



    def get_threshold_masks(self):
        """
        return a thresholder version of the input grayscale mask where the mask is positive for points that are greater
        than the given threshold
        :return:
        """
        self.threshold_masks_path = data_functions.create_path(self.thres_save_path, f'binary')
        for img in self.image_obj_iterator():
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
            image_cut=img.cut_via_mask()
            cv2.imwrite(os.path.join(self.cut_image_path, img.image_name), image_cut)

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

    @staticmethod
    def get_label(pr_green, pr_brown,img_name):
        label = 'D'
        if pr_green >= 0.5:
            if pr_brown < 0.1:
                label = 'A'
            if pr_brown > 0.1:
                label = 'B'
        elif 0.5 > pr_green > 0.1:
            p = pr_brown/pr_green
            if p < 0.25:
                label = 'A'
            elif 0.25 <= p < 0.6:
                label = 'B'
            else:
                label = 'C'
        x=label
        return label

    """    def get_label(pr_green,pr_brown):
        label = 'D'
        if pr_green >= 0.5 and pr_brown < 0.2: label ='A'
        elif  0.5 > pr_green >= 0.3 and 0.3 >pr_brown >= 0.2: label = 'B'
        elif   0.3 > pr_green and 0.4 > pr_brown >= 0.3: label = 'C'
        return label"""

    @staticmethod
    def get_label1(pr):
        label = 'A'
        if pr >= 0.5: label ='D'
        elif 0.5 > pr >=0.3: label = 'C'
        elif 0.3 > pr >=0.1: label = 'B'
        return label

    def fillter_via_color(self,save_flag=False):
        logger.debug(" <- fillter_via_color")
        out_path = data_functions.create_path(self.thres_save_path, f'filtered')
        logger.info(f"creting filltered images in {out_path}")
        for img in tqdm(self.image_obj_iterator()):
            pr_green,pr_brown = img.filter_cut_iamge()
            raw_name = img.image_name.rsplit('.',1)[0]
            pred = self.get_label(pr_green,pr_brown,img.image_name)
            curr_save_path = data_functions.create_path(out_path, pred)
            _ = shutil.copy(img.img_path, curr_save_path)
            if save_flag:
                cv2.imwrite(os.path.join(out_path, f'{raw_name}_green.jpg'), img.green_part)
                cv2.imwrite(os.path.join(out_path, f'{raw_name}_brown.jpg'), img.brown_part)
                cv2.imwrite(os.path.join(out_path, img.image_name), img.threshold_mask)
        logger.debug(" -> get_notop")



    def image_obj_iterator(self):
        for img_entry in os.scandir(self.img_path):
            mask_name = img_entry.name.rsplit(".",1)[0]+'.npy'
            mask_path = os.path.join(self.mask_path, mask_name)
            img = CustomImage(img_entry.path, mask_path,threshold=self.threshold)
            yield img


    def calc_hists(self,hist_type='brg'):
        dest_path = data_functions.create_path(self.thres_save_path, f'{hist_type}_histograms')

        for img in tqdm(self.image_obj_iterator()):
            img_raw_name = img.image_name.rsplit('.',1)[0]
            curr_dest_path = os.path.join(dest_path,f"{img_raw_name}.npy")
            fig_big_hist= img.get_hist_via_mask(hist_type=hist_type)
            np.save(curr_dest_path,fig_big_hist)


def load_npy_data(src_path):
    df = None
    name_list = []
    for i, file_entry in enumerate(os.scandir(src_path)):
        if file_entry.name.endswith('.npy'):
            file = normalize(np.load(file_entry.path)).flatten()
            name = file_entry.name.rsplit('.', 1)[0]
            name_list.append(name)
            if df is None:
                df = pd.DataFrame(file)
            else:
                df[i] = file

    df = df.T
    df.columns = df.columns.astype(str)
    df.insert(0, "file_name", name_list)

    return df

def load_data(path,hist_type):
    logger.debug(" <- load_data")
    logger.debug(f"loading train data from:{path}")
    ret_df = pd.DataFrame()

    for label_folder in os.scandir(path):
        hist_folder = os.path.join(label_folder.path, f'{hist_type}_histograms')
        curr_df = load_npy_data(hist_folder)
        curr_df['label'] =label_folder.name

        ret_df = pd.concat((ret_df,curr_df))

    ret_df['label'] = ret_df['label'].astype('category')
    ret_df = shuffle(ret_df)
    ret_df.reset_index(inplace=True,drop=True)

    logger.debug(" -> load_data")

    return ret_df














