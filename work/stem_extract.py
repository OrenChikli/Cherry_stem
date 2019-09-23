from work.preprocess import display_functions,data_functions
import os
import cv2
import numpy as np

from work.segmentation.clarifruit_segmentation.image import Image


class StemExtractor:

    def __init__(self,clasess_path,img_path,mask_path,stems_path=None,cut_mask_path=None):

        self.img_path = img_path
        self.mask_path = mask_path
        self.stems_path = stems_path
        self.classes_dict = None
        self.get_clases_dict(clasess_path)
        self.cut_mask_path = cut_mask_path

    def get_clases_dict(self,classes_path):
        self.classes_dict = {}
        for img_entry in os.scandir(classes_path):
            img_color = cv2.imread(img_entry.path, cv2.IMREAD_COLOR)
            img_color = img_color[0, 0]
            class_name = img_entry.name.split('.')[0]
            self.classes_dict[class_name] = img_color


    def get_stems(self):
        for img_entry in os.scandir(self.img_path):
            mask_path = os.path.join(self.mask_path, img_entry.name)
            img = Image(img_entry.path, mask_path)
            img.cut_via_mask(save_flag=True,dest_path=self.stems_path)

    def get_mean_h(self,save_path):
        for img_entry in os.scandir(self.img_path):
            mask_path = os.path.join(self.mask_path, img_entry.name)
            cut_mask_path = os.path.join(self.cut_mask_path,img_entry.name)
            img = Image(img_entry.path, mask_path, cut_mask_path=cut_mask_path)
            img.get_mean_hue(save_flag=True,dest_path=save_path)

    def get_mean_color(self,save_path):
        for img_entry in os.scandir(self.img_path):
            mask_path = os.path.join(self.mask_path, img_entry.name)
            cut_mask_path = os.path.join(self.cut_mask_path,img_entry.name)
            img = Image(img_entry.path, mask_path, cut_mask_path=cut_mask_path)
            img.get_mean_color(save_flag=True,dest_path=save_path)


    def score_stem_color(self,img_path,dest_path):

        res_dict = dict()
        for img_entry in os.scandir(img_path):

            img_color = cv2.imread(img_entry.path, cv2.IMREAD_COLOR)
            img_color = img_color[0, 0]
            res = data_functions.find_closest(img_color, self.classes_dict)
            res_dict[img_entry.name] = res
            data_functions.save_json(res_dict, "image_classifications_hsv.json", dest_path)









def main():
    classes_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\hsv'
    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'

    curr_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_23-55-20'

    mask_path = data_functions.create_path(curr_path,'binary_thres_0.3')

    stems_path = data_functions.create_path(curr_path,'stems')
    class_dest_path =data_functions.create_path(curr_path,'classification')
    mean_h_dest_path= data_functions.create_path(curr_path,'mean_h')
    mean_color_dest_path = data_functions.create_path(curr_path, 'color')

    mean_h_dest_path = data_functions.create_path(mean_h_dest_path,'via_thres_0.3')
    mean_color_dest_path = data_functions.create_path(mean_color_dest_path,'via_0.3')
    stems_path = data_functions.create_path(stems_path, 'via_0.3')


    stem_exctractor = StemExtractor(classes_path,img_path,mask_path,stems_path,stems_path)
    # stem_exctractor.get_stems()
    # stem_exctractor.get_mean_h(mean_h_dest_path)
    # stem_exctractor.get_mean_color(mean_color_dest_path)
    stem_exctractor.score_stem_color(mean_color_dest_path,class_dest_path)


if __name__ == '__main__':
    main()