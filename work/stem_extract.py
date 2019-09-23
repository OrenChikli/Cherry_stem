from work.preprocess import display_functions,data_functions
import os
import cv2
import numpy as np
from scipy.stats import circmean

from work.segmentation.clarifruit_segmentation.image import Image


class StemExtractor:

    def __init__(self,clasess_path,img_path,mask_path):

        self.img_path = img_path
        self.mask_path = mask_path
        self.classes_dict = None
        self.get_clases_dict(clasess_path)


    def get_clases_dict(self,classes_path):
        self.classes_dict = {}
        for img_entry in os.scandir(classes_path):
            img_color = cv2.imread(img_entry.path, cv2.IMREAD_COLOR)
            img_color = img_color[0, 0]
            class_name = img_entry.name.split('.')[0]
            self.classes_dict[class_name] = img_color

    # def threshold_mask(self,threshold,save_path):
    #     for img_entry in os.scandir(self.img_path):
    #         mask_path = os.path.join(self.mask_path, img_entry.name)
    #         img = Image(img_entry.path, mask_path,threshold=threshold)
    #         cv2.imwrite(os.path.join(save_path, img.image_name), img.threshold_mask)

    def threshold_mask(self,threshold,save_path):
        for img in self.image_obj_iterator(threshold=threshold):
            cv2.imwrite(os.path.join(save_path, img.image_name), img.threshold_mask)


    # def get_stems(self, save_path):
    #     for img_entry in os.scandir(self.img_path):
    #         mask_path = os.path.join(self.mask_path, img_entry.name)
    #         img = Image(img_entry.path, mask_path)
    #         img.cut_via_mask(save_flag=True, dest_path=save_path)

    def get_stems(self, save_path):
        for img in self.image_obj_iterator():
            img.cut_via_mask(save_flag=True, dest_path=save_path)

    # def get_mean_h(self, save_path):
    #     for img_entry in os.scandir(self.img_path):
    #         mask_path = os.path.join(self.mask_path, img_entry.name)
    #         img = Image(img_entry.path, mask_path)
    #         img.get_mean_hue(save_flag=True, dest_path=save_path)

    def get_mean_h(self, save_path):
        for img in self.image_obj_iterator():
            img.get_mean_hue(save_flag=True, dest_path=save_path)


    # def get_mean_color(self,save_path):
    #     for img_entry in os.scandir(self.img_path):
    #         mask_path = os.path.join(self.mask_path, img_entry.name)
    #         img = Image(img_entry.path, mask_path)
    #
    #         img.get_mean_color(save_flag=True,dest_path=save_path)

    def get_mean_color(self, save_path):
        for img in self.image_obj_iterator():
            img.get_mean_color(save_flag=True, dest_path=save_path)

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

    def sharpen_maskes(self,save_path):
        for img_entry in os.scandir(self.img_path):
            mask_path = os.path.join(self.mask_path, img_entry.name)
            img = Image(img_entry.path, mask_path)
            img.sharpen_mask(save_flag=True,dest_path=save_path)

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


    # @staticmethod
    # def get_mean_h(src_path, save_flag=False, dest_path=None):
    #     for img_entry in os.scandir(src_path):
    #         mask= cv2.imread(img_entry.path,cv2.IMREAD_COLOR)
    #         hsv = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
    #         non_zero= np.any(mask != [0, 0, 0], axis=-1)
    #         h = hsv[:, :, 0]
    #         h_nonzero = h[non_zero]
    #         h_rad = np.deg2rad(h_nonzero)
    #         mean_h = np.rad2deg(circmean(h_rad))
    #         hsv = ((mean_h, 255, 255) * np.ones_like(mask)).astype(np.uint8)
    #         res_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    #         if save_flag:
    #             cv2.imwrite(os.path.join(dest_path, img_entry.name), res_img)




def main():
    classes_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\hsv'
    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'

    curr_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_23-55-20'

    mask_path = data_functions.create_path(curr_path,'raw_pred')

    threshold=0.5
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
    mean_color_dest_path = data_functions.create_path(mean_color_dest_path,'sharpened')
    stems_path = data_functions.create_path(stems_path, 'sharpened')

    x = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\image'
    y = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\label'
    z = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\New folder'
    w=r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\imge_masked'

    stem_exctractor = StemExtractor(classes_path,x,y)
    #stem_exctractor.threshold_mask(0.5,w)
    #stem_exctractor.sharpen_maskes(sharpened_mask_path)
    stem_exctractor.get_stems(w)

    stem_exctractor.get_mean_h(w, True, z)
    #stem_exctractor.get_mean_h(mean_h_dest_path)
    #stem_exctractor.get_mean_color(mean_color_dest_path)
    #stem_exctractor.score_stem_color(mean_h_dest_path,class_dest_path)


if __name__ == '__main__':
    main()
