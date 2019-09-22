from work.preprocess import display_functions,data_functions
import os
import cv2
import numpy as np

from scipy.stats import circmean
from work.segmentation.clarifruit_segmentation.image import Image


class StemExtractor:

    def __init__(self,clasess_path,img_path,mask_path,stems_path=None):

        self.img_path = img_path
        self.mask_path = mask_path
        self.stems_path = stems_path
        self.classes_dict = None
        self.get_clases_dict(clasess_path)

    def get_clases_dict(self,classes_path):
        self.classes_dict = {}
        for img_entry in os.scandir(classes_path):
            img_color = cv2.imread(img_entry.path, cv2.IMREAD_COLOR)
            img_color = img_color[0, 0]
            class_name = img_entry.name.split('.')[0]
            self.classes_dict[class_name] = img_color


    def get_stems(self,save_path):
        for img_entry in os.scandir(self.img_path):
            mask_path = os.path.join(self.mask_path, img_entry.name)
            img = Image(img_entry.path, mask_path)
            img.cut_via_mask(save_flag=True,dest_path=save_path)

    def get_mean_h(self,save_path):
        for img_entry in os.scandir(self.img_path):
            mask_path = os.path.join(self.mask_path, img_entry.name)
            img = Image(img_entry.path, mask_path)
            img.get_mean_hue(save_flag=False,dest_path=save_path)


    def score_stem_color(self,dest_path):

        res_dict = dict()
        for img_entry in os.scandir(self.img_path):

            img_color = cv2.imread(img_entry.path, cv2.IMREAD_COLOR)
            img_color = img_color[0, 0]
            res = data_functions.find_closest(img_color, self.classes_dict)
            res_dict[img_entry.name] = res
            data_functions.save_json(res_dict, "image_classifications_hsv.json", dest_path)




def get_avg_patch(img,mask):
    out = cv2.mean(img, mask)[:-1]
    avg_patch = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(out)
    return avg_patch

def use_funcs():
    src_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    mask_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-21_22-30-30\binary_thres_0.5'
    dest_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-21_22-30-30\mean_color'
    apply_func(get_avg_patch,src_path,mask_path,dest_path)


def apply_func(func,src_path,mask_path,dest_path):
    for img_entry in os.scandir(src_path):
        img = cv2.imread(img_entry.path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(os.path.join(mask_path, img_entry.name), cv2.IMREAD_UNCHANGED)
        res = func(img,mask)
        cv2.imwrite(os.path.join(dest_path, img_entry.name), res)


def use_get_average_color():
    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    mask_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\binary_thres_0.5'
    dest_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\color'

    get_average_color(img_path, mask_path, dest_path)

def get_average_color(img_path,mask_path,dest_path):

    for img_entry in os.scandir(img_path):

        img = cv2.imread(img_entry.path,cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(mask_path,img_entry.name),cv2.IMREAD_UNCHANGED)
        out = cv2.mean(img,mask)[:-1]
        avg_patch = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(out)
        cv2.imwrite(os.path.join(dest_path, img_entry.name), avg_patch)




def main():
    classes_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\hsv'
    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    mask_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\raw_pred'
    stems_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\stems_from_raw'

    dest_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\classification'

    stem_exctractor = StemExtractor(classes_path,img_path,mask_path,stems_path)
    stem_exctractor.score_stem_color(dest_path)
    #get_stems()
    #get_average_color()
    #use_get_average_color()

    #get_mean_hue()

if __name__ == '__main__':
    main()