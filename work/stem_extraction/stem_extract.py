from work.preprocess import data_functions
import os
import cv2

import shutil

from work.logger_settings import *

configure_logger()
logger = logging.getLogger(__name__)

from work.segmentation.clarifruit_segmentation.image import Image


class StemExtractor:

    def __init__(self,img_path,mask_path,h_clasess_path,color_classes_path,threshold=125):

        self.img_path = img_path
        self.threshold = threshold
        self.save_path = mask_path
        self.thres_save_path = data_functions.create_path(mask_path, f"thres_{threshold}")
        self.raw_mask_path = os.path.join(mask_path,'raw_masks')


        self.threshold_masks_path = None

        self.sharp_masks_path = None
        self.cut_image_path = None
        self.mean_h_path = None
        self.mean_color_path=None
        self.h_classes_dict = self.get_clases_dict(h_clasess_path)
        self.color_classes_dict =self.get_clases_dict(color_classes_path)



    @staticmethod
    def get_clases_dict(classes_path):
        classes_dict = {}
        for img_entry in os.scandir(classes_path):
            img_color = cv2.imread(img_entry.path, cv2.IMREAD_COLOR)
            img_color = img_color[0, 0]
            class_name = img_entry.name.split('.')[0]
            classes_dict[class_name] = img_color
        return classes_dict


    def get_threshold_masks(self,type_flag='orig'):
        if type_flag=='orig':
            self.threshold_masks_path = data_functions.create_path(self.thres_save_path, f'binary')
        elif type_flag=='sharp':
            self.threshold_masks_path = data_functions.create_path(self.thres_save_path, f'sharp_binary')
        for img in self.image_obj_iterator():
            cv2.imwrite(os.path.join(self.threshold_masks_path, img.image_name), img.threshold_mask)

    def sharpen_maskes(self):
        self.sharp_masks_path = data_functions.create_path(self.save_path, 'sharpened')
        for img in self.image_obj_iterator():
            img.get_sharp_mask()
            cv2.imwrite(os.path.join(self.sharp_masks_path, img.image_name), img.grayscale_mask)

    def get_stems(self,type_flag='orig'):
        if type_flag == 'orig':
            self.cut_image_path = data_functions.create_path(self.thres_save_path, f'orig_stems')
            for img in self.image_obj_iterator():
                img.cut_via_mask()
                cv2.imwrite(os.path.join(self.cut_image_path, img.image_name), img.image_cut)

        if type_flag == 'sharp':
            self.cut_image_path = data_functions.create_path(self.thres_save_path, f'sharp_stems')
            for img in self.image_obj_iterator():
                img.get_sharp_mask()
                img.cut_via_mask()
                cv2.imwrite(os.path.join(self.cut_image_path, img.image_name), img.grayscale_mask)


    def get_mean_h(self,type_flag='orig'):
        if type_flag == 'orig':
            self.mean_h_path = data_functions.create_path(self.thres_save_path, f'orig_mean_h')
            for img in self.image_obj_iterator():
                img.get_mean_hue()
                cv2.imwrite(os.path.join(self.mean_h_path, img.image_name), img.mask_mean_h)

        if type_flag == 'sharp':
            self.mean_h_path = data_functions.create_path(self.thres_save_path, f'sharp_mean_h')
            for img in self.image_obj_iterator():
                img.get_sharp_mask()
                img.get_mean_hue()
                cv2.imwrite(os.path.join(self.mean_h_path, img.image_name), img.mask_mean_h)

    def get_mean_color(self, type_flag='orig'):
        if type_flag == 'orig':
            self.mean_color_path = data_functions.create_path(self.thres_save_path, f'orig_mean_color')
            for img in self.image_obj_iterator():
                img.get_mean_color()
                cv2.imwrite(os.path.join(self.mean_color_path, img.image_name), img.mask_mean_color)

        if type_flag == 'sharp':
            self.mean_color_path = data_functions.create_path(self.thres_save_path, f'sharp_mean_color')
            for img in self.image_obj_iterator():
                img.get_sharp_mask()
                img.get_mean_color()
                cv2.imwrite(os.path.join(self.mean_color_path, img.image_name), img.mask_mean_color)



    def image_obj_iterator(self):
        for img_entry in os.scandir(self.img_path):
            mask_path = os.path.join(self.raw_mask_path, img_entry.name)
            img = Image(img_entry.path, mask_path,threshold=self.threshold)
            yield img



    def score_results(self,score_type='heu',mask_type='orig'):

        #res_dict = dict()
        if score_type=='heu':
            src_path = self.mean_h_path
            dest_path = data_functions.create_path(self.thres_save_path,f'mean_h_{mask_type}_scores')
            curr_classes_dict = self.h_classes_dict
        elif score_type =='color':
            src_path = self.mean_color_path
            dest_path = data_functions.create_path(self.thres_save_path,f'mean_color_{mask_type}_scores')
            curr_classes_dict = self.color_classes_dict
        else:
            message = "wrong_flag_type"
            logger.info("wrong_flag_type")
            raise IOError(message)


        for img_class,_ in curr_classes_dict.items():
            data_functions.create_path(dest_path,img_class)

        for img_entry in os.scandir(src_path):

            img_color = cv2.imread(img_entry.path, cv2.IMREAD_COLOR)
            img_color = img_color[0, 0]
            res = data_functions.find_closest(img_color, curr_classes_dict)
            #res_dict[img_entry.name] = res
            curr_dest_path = os.path.join(dest_path,res)
            _ = shutil.copy(img_entry.path, curr_dest_path)






