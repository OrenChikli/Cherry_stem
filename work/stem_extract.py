from work.preprocess import display_functions,data_functions
import os
import cv2
import numpy as np
from scipy.stats import circmean

from work.segmentation.clarifruit_segmentation.image import Image


class StemExtractor:

    def __init__(self,clasess_path,img_path,mask_path,threshold=125):

        self.img_path = img_path
        self.masks_path = mask_path
        self.raw_mask_path = os.path.join(mask_path,'raw_masks')

        self.threshold=threshold
        self.threshold_masks_path = None

        self.sharp_masks_path = None
        self.cut_image_path = None
        self.classes_dict = None
        self.get_clases_dict(clasess_path)



    def get_clases_dict(self,classes_path):
        self.classes_dict = {}
        for img_entry in os.scandir(classes_path):
            img_color = cv2.imread(img_entry.path, cv2.IMREAD_COLOR)
            img_color = img_color[0, 0]
            class_name = img_entry.name.split('.')[0]
            self.classes_dict[class_name] = img_color


    def get_threshold_masks(self,threshold=125):
        self.threshold=threshold
        self.threshold_masks_path = data_functions.create_path(self.masks_path,f'binary_{threshold}')

        for img in self.image_obj_iterator(threshold=threshold):
            img.get_threshold_mask()
            cv2.imwrite(os.path.join(self.threshold_masks_path, img.image_name), img.threshold_mask)

    def sharpen_maskes(self):
        self.sharp_masks_path = data_functions.create_path(self.masks_path, 'sharpened')
        for img in self.image_obj_iterator():
            cv2.imwrite(os.path.join(self.shap_masks_path, img.image_name), img.sharp_mask)

    def get_stems(self,type_flag='orig', threshold=125):
        if type_flag == 'orig':
            self.cut_image_path = data_functions.create_path(self.masks_path,f'orig_cut_thres{threshold}')
            for img in self.image_obj_iterator(threshold=threshold):
                img.cut_via_mask()
                cv2.imwrite(os.path.join(self.cut_image_path, img.image_name), img.sharp_mask)

        if type_flag == 'sharp':
            self.cut_image_path = data_functions.create_path(self.masks_path,f'sharp_cut_thres{threshold}')
            for img in self.image_obj_iterator(threshold=threshold):
                img.cut_via_mask()
                cv2.imwrite(os.path.join(self.cut_image_path, img.image_name), img.sharp_mask)


    def get_mean_h(self):
        for img in self.image_obj_iterator(threshold=threshold):
            img.get_mean_hue()
            cv2.imwrite(os.path.join(self.cut_image_path, img.image_name), img.sharp_mask)


    def get_mean_color(self, save_path):
        color_dict = dict()
        for img in self.image_obj_iterator():
            res = img.get_mean_color(save_flag=True, dest_path=save_path)
            color_dict[img.image_name] = res
        data_functions.save_json(color_dict,'color_labels.json',save_path)

    def image_obj_iterator(self,**kwargs):
        for img_entry in os.scandir(self.img_path):
            mask_path = os.path.join(self.mask_path, img_entry.name)
            img = Image(img_entry.path, mask_path,**kwargs)
            yield img


    def score_stem_color(self,img_path,dest_path):

        res_dict = dict()
        for img_entry in os.scandir(img_path):

            img_color = cv2.imread(img_entry.path, cv2.IMREAD_COLOR)
            img_color = img_color[0, 0]
            res = data_functions.find_closest(img_color, self.classes_dict)
            res_dict[img_entry.name] = res
            data_functions.save_json(res_dict, "image_classifications_hsv.json", dest_path)



    @staticmethod
    def get_mean_h(src_path, save_flag=False, dest_path=None):
        for img_entry in os.scandir(src_path):
            stem = cv2.imread(img_entry.path,cv2.IMREAD_COLOR)
            hsv = cv2.cvtColor(stem, cv2.COLOR_RGB2HSV)
            non_zero= np.any(stem != [0, 0, 0], axis=-1)
            h = hsv[:, :, 0]
            h_nonzero = h[non_zero]
            h_rad = np.deg2rad(h_nonzero)
            mean_h = np.rad2deg(circmean(h_rad))
            hsv = ((mean_h, 255, 255) * np.ones_like(stem)).astype(np.uint8)
            res_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            if save_flag:
                cv2.imwrite(os.path.join(dest_path, img_entry.name), res_img)


def main():
    classes_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\hsv'
    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'

    curr_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_23-55-20'

    mask_path = data_functions.create_path(curr_path,'raw_pred')

    threshold=0.4
    thres_mask_path =data_functions.create_path(curr_path,f"binary_{threshold}")
    thres_sharp_mask_path = data_functions.create_path(curr_path, f"sharp_binary_{threshold}")


    sharpened_mask_path = data_functions.create_path(curr_path,'pred_shrap')


    stems_path = data_functions.create_path(curr_path,'stems')
    stems_raw = data_functions.create_path(stems_path,'raw_mask')
    stems_sharp = data_functions.create_path(stems_path,'sharp_mask')
    stems_sharp_bin = data_functions.create_path(stems_path, 'sharp_binary_0.5_mask')

    class_dest_path =data_functions.create_path(curr_path,'classification')
    mean_h_dest_path= data_functions.create_path(curr_path,'mean_h')
    mean_color_dest_path = data_functions.create_path(curr_path, 'color')

    mean_h_dest_path = data_functions.create_path(mean_h_dest_path,'sharpened')
    mean_color_dest_path = data_functions.create_path(mean_color_dest_path,'binary_0.5')
    stems_path = data_functions.create_path(stems_path, 'binary_0.5')

    stem_exctractor = StemExtractor(classes_path,img_path,mask_path)
    stem_exctractor.threshold_mask(threshold,thres_mask_path)
    #stem_exctractor.sharpen_maskes(sharpened_mask_path)
    #stem_exctractor.get_stems(stems_path)
    #stem_exctractor.get_mean_color(mean_color_dest_path)
    #stem_exctractor.get_mean_h(w, True, z)
    #stem_exctractor.get_mean_h(mean_h_dest_path)
    #stem_exctractor.get_mean_color(mean_color_dest_path)
    #stem_exctractor.score_stem_color(mean_h_dest_path,class_dest_path)


if __name__ == '__main__':
    main()
