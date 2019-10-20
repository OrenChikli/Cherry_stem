from work.auxiliary import data_functions
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

import logging
from work.auxiliary import decorators
from work.auxiliary.exceptions import *


logger = logging.getLogger(__name__)
logger_decorator = decorators.Logger_decorator(logger)

from work.auxiliary.custom_image import CustomImage
from work.segmentation.segmentation import SegmentationSingle

OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Chi-Squared", cv2.HISTCMP_CHISQR),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

HIST_TYPE = {'rgb': lambda x: x.get_hist_via_mask,
             'hsv': lambda x: x.get_hsv_hist}

FUNC_DICTIPNARY = {'binary': lambda x: x.get_threshold_masks(),
                   'hists': lambda x: x.calc_hists(),
                   'ontop': lambda x: x.get_ontop_images(),
                   'stems': lambda x: x.get_stems(),
                   'filter_images': lambda x: x.filter_images()}



class StemExtractor:
    """
        get color histograms of the source images using the unet maskes, i.e get color histograms of the
    areas activated in the binary mask
    :param img_path: path to source images
    :param mask_path:path to the unet masks of the source images
    :param save_path: the save path of the results
    :param threshold: the threshold to get the binary mask, the binary mask will be activated for pixels that have
    values greater than the threshold
    :param hist_type: the type of histogram to calculate. default value is for 'bgr' which is the normal color mode.
    another option is 'hsv' for heu,saturation and value color space
    :return:
    """

    @logger_decorator.debug_dec
    def __init__(self, img_path, mask_path, save_path, threshold=0.5,
                 use_thres_flag=True,hist_type='bgr'):
        """

        :param img_path: path to source images
        :param mask_path:path to the unet masks of the source images
        :param save_path: the save path of the results
        :param threshold: the threshold to get the binary mask, the binary mask will be activated for pixels that have
        values greater than the threshold

        :param use_thres_flag:
        :param hist_type: the type of histogram to calculate. default value is for 'bgr' which is the normal color mode.
        another option is 'hsv' for heu,saturation and value color space
        """

        self.img_path = img_path
        self.mask_path = mask_path
        self.threshold = threshold
        if use_thres_flag:
            self.thres_save_path = data_functions.create_path(save_path,
                                                              f"thres_{threshold}")
        else:
            self.thres_save_path = save_path

        self.threshold_masks_path = None

        self.cut_image_path = None
        self.ontop_path = None
        self.hist_type=hist_type

    @logger_decorator.debug_dec
    def get_threshold_masks(self):
        """
        return a thresholder version of the input grayscale mask where the mask is positive for points that are greater
        than the given threshold
        :return:
        """
        logger.info(
            f"getting binary masks with threshold: {self.threshold} "
            f"from{self.mask_path}")

        self.threshold_masks_path = data_functions.create_path(
            self.thres_save_path, f'binary')

        logger.info(f"saving results at {self.threshold_masks_path}")
        for img in self.image_obj_iterator():
            img_save_path = os.path.join(self.threshold_masks_path,
                                         img.image_name)
            cv2.imwrite(img_save_path, img.threshold_mask)

    @logger_decorator.debug_dec
    def get_stems(self):
        """
        extract the "stems" from the image - return the areas in the image that are activated in the thresholded mask
        eg if pixel (156,46) is turned on in the mask image, it will show in the result
        :return:
        """

        self.cut_image_path = data_functions.create_path(self.thres_save_path,
                                                         f'stems')
        logger.info(f"getting stems for threshold {self.threshold}")
        logger.info(f"creting stems in {self.cut_image_path}")

        for img in self.image_obj_iterator():
            image_cut = img.cut_via_mask()
            cv2.imwrite(os.path.join(self.cut_image_path, img.image_name),
                        image_cut)

    @logger_decorator.debug_dec
    def get_ontop_images(self):
        """
        return the overlay of the thresholded mask on top of the source image
        :return:
        """

        self.ontop_path = data_functions.create_path(self.thres_save_path,
                                                     f'on_top')
        logger.info(f"getting ontop images for threshold {self.threshold}")
        logger.info(f"creting ontop images in {self.ontop_path}")
        for img in self.image_obj_iterator():
            cv2.imwrite(os.path.join(self.ontop_path, img.img_name),
                        img.get_ontop())



    @logger_decorator.debug_dec
    @staticmethod
    def get_label(pr_green, pr_brown, img_name):
        label = 'D'
        if pr_green >= 0.5:
            if pr_brown < 0.1:
                label = 'A'
            if pr_brown > 0.1:
                label = 'B'
        elif 0.5 > pr_green > 0.1:
            p = pr_brown / pr_green
            if p < 0.25:
                label = 'A'
            elif 0.25 <= p < 0.6:
                label = 'B'
            else:
                label = 'C'
        x = label
        return label

    """    def get_label(pr_green,pr_brown):
        label = 'D'
        if pr_green >= 0.5 and pr_brown < 0.2: label ='A'
        elif  0.5 > pr_green >= 0.3 and 0.3 >pr_brown >= 0.2: label = 'B'
        elif   0.3 > pr_green and 0.4 > pr_brown >= 0.3: label = 'C'
        return label"""

    @logger_decorator.debug_dec
    @staticmethod
    def get_label1(pr):
        label = 'A'
        if pr >= 0.5:
            label = 'D'
        elif 0.5 > pr >= 0.3:
            label = 'C'
        elif 0.3 > pr >= 0.1:
            label = 'B'
        return label

    # def fillter_via_color(self,save_flag=False):
    #     logger.debug(" <- fillter_via_color")
    #     out_path = data_functions.create_path(self.thres_save_path, f'filtered')
    #     logger.info(f"creting filltered images in {out_path}")
    #     for img in tqdm(self.image_obj_iterator()):
    #         pr_green,pr_brown = img.filter_cut_iamge()
    #         raw_name = img.image_name.rsplit('.',1)[0]
    #         pred = self.get_label(pr_green,pr_brown,img.image_name)
    #         curr_save_path = data_functions.create_path(out_path, pred)
    #         _ = shutil.copy(img.img_path, curr_save_path)
    #         if save_flag:
    #             cv2.imwrite(os.path.join(out_path, f'{raw_name}_green.jpg'), img.green_part)
    #             cv2.imwrite(os.path.join(out_path, f'{raw_name}_brown.jpg'), img.brown_part)
    #             cv2.imwrite(os.path.join(out_path, img.image_name), img.threshold_mask)
    #     logger.debug(" -> get_notop")

    @logger_decorator.debug_dec
    def fillter_via_color_green_brown(self, save_flag=False):
        out_path = data_functions.create_path(self.thres_save_path, f'filtered')
        logger.info(f"creting filltered images in {out_path}")
        for img in self.image_obj_iterator():
            pr_green, pr_brown = img.filter_cut_image_green_brown()
            raw_name = img.image_name.rsplit('.', 1)[0]
            pred = self.get_label(pr_green, pr_brown, img.image_name)
            curr_save_path = data_functions.create_path(out_path, pred)
            _ = shutil.copy(img.img_path, curr_save_path)
            if save_flag:
                cv2.imwrite(os.path.join(out_path, f'{raw_name}_green.jpg'),
                            img.green_part)
                cv2.imwrite(os.path.join(out_path, f'{raw_name}_brown.jpg'),
                            img.brown_part)
                cv2.imwrite(os.path.join(out_path, img.image_name),
                            img.threshold_mask)

    @logger_decorator.debug_dec
    def fillter_via_color(self):

        out_path = data_functions.create_path(self.thres_save_path, f'filtered')
        logger.info(f"getting filterred images for threshold {self.threshold}")
        logger.info(f"creting filltered images in {out_path}")
        for img in self.image_obj_iterator():
            res = img.filter_cut_image()
            cv2.imwrite(os.path.join(out_path, img.image_name), res)

    @logger_decorator.debug_dec
    def image_obj_iterator(self):
        for img_entry in os.scandir(self.img_path):
            mask_name = img_entry.name.rsplit(".", 1)[0] + '.npy'
            mask_path = os.path.join(self.mask_path, mask_name)
            img = CustomImage(img_path=img_entry.path,
                              mask_path=mask_path,
                              is_binary_mask=False,
                              threshold=self.threshold,
                              create_save_dest_flag=False)

            yield img

    @logger_decorator.debug_dec
    def calc_hists(self):
        dest_path = data_functions.create_path(self.thres_save_path,
                                               f'{self.hist_type}_histograms')
        logger.info(f"getting {self.hist_type} histograms for threshold {self.threshold}")
        logger.info(f"saving results at {dest_path}")
        for img in self.image_obj_iterator():
            curr_dest_path = os.path.join(dest_path, f"{img.img_raw_name}.npy")
            fig_big_hist = img.get_hist_via_mask(hist_type=self.hist_type)
            np.save(curr_dest_path, fig_big_hist)




@logger_decorator.debug_dec
def create_object(img_path, mask_path, save_path, threshold, hist_type,
                  use_thres_flag,
                  obj_type):
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    save_path=save_path,
                                    threshold=threshold,
                                    use_thres_flag=use_thres_flag,
                                    hist_type=hist_type)

    FUNC_DICTIPNARY[obj_type](stem_exctractor)

@logger_decorator.debug_dec
def create_ground_truth_objects(ground_path, threshold, src_path, obj_type,
                                folder='classifier_train_data',
                                hist_type='bgr'):
    dest_path = data_functions.create_path(src_path, f"thres_{threshold}")
    dest_path = data_functions.create_path(dest_path, folder)
    raw_pred_path = os.path.join(src_path, folder)
    ground_path = os.path.join(ground_path,folder)
    logger.info(f"getting {obj_type} objects for {folder}")
    for curr_class in os.scandir(raw_pred_path):
        logger.info(f"getting objects for {curr_class.name} class")
        curr_raw_pred_path = os.path.join(raw_pred_path, curr_class.name)
        curr_dest = data_functions.create_path(dest_path, curr_class.name)
        curr_ground_path = os.path.join(ground_path, curr_class.name)

        create_object(img_path=curr_ground_path,
                      mask_path=curr_raw_pred_path,
                      save_path=curr_dest,
                      threshold=threshold,
                      hist_type=hist_type,
                      use_thres_flag=False,
                      obj_type=obj_type)


def create_test_train_obj(ground_path,threshold,src_path,obj_type,hist_type):
    create_ground_truth_objects(ground_path, threshold, src_path, obj_type,
                                folder='train',
                                hist_type=hist_type)
    create_ground_truth_objects(ground_path, threshold, src_path, obj_type,
                                folder='test',
                                hist_type=hist_type)