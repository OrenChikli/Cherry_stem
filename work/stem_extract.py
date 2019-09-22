from work.preprocess import display_functions,data_functions
import os
import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import circmean




class StemExtractor:

    def __init__(self,imgs_path,mask_path,dest_path):
        self.imgs_path = imgs_path
        self.masks_path = mask_path
        self.dest_path = dest_path


def get_mean_hue():
    src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\stems_from_raw'
    dest_path= r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\mean_h_from_raw'

    for img_entry in os.scandir(src_path):
        img = cv2.imread(img_entry.path, cv2.IMREAD_UNCHANGED)
        x = img > 0
        img_not_zero = img[img > 0]
        img_hsv=  cv2.cvtColor(img, cv2.COLOR_RGB2HSV) * (img > 0)
        img_h = img_hsv[:,:,0]
        img_h = img_h[img_h > 0]
        img_h_rad = np.deg2rad(img_h)
        mean_hue = np.rad2deg(circmean(img_h_rad))
        hsv = ((mean_hue,255,255)*np.ones_like(img)).astype(np.uint8)
        res_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        cv2.imwrite(os.path.join(dest_path,img_entry.name),res_img)



def get_stems():
    img_path=r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    mask_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\raw_pred'
    dest_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\stems_from_raw'
    display_functions.cut_via_mask(img_path,mask_path,dest_path)

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



def find_closest(color,colors_dict):

    distances = [(euclidean(color, curr_color), label) for label,curr_color in colors_dict.items()]
    max = sorted(distances)
    best = max[0][1]
    return best



def score_stem_color(stems_path,scores_path,res_path):
    classes_dict = dict()
    res_dict = dict()

    for img_entry in os.scandir(scores_path):
        img_color = cv2.imread(img_entry.path,cv2.IMREAD_COLOR)
        img_color=img_color[0,0]
        class_name = img_entry.name.split('.')[0]
        classes_dict[class_name] = img_color

    for img_entry in os.scandir(stems_path):
        img_color = cv2.imread(img_entry.path,cv2.IMREAD_COLOR)
        img_color=img_color[0,0]
        res = find_closest(img_color,classes_dict)
        res_dict[img_entry.name] = res

        data_functions.save_json(res_dict, "image_classifications_hsv.json", res_path)


def get_stem_scores():
    stems_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\mean_h_from_raw'
    classes_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\hsv'
    res_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\classification'
    score_stem_color(stems_path,classes_path,res_path)


def get_color_mask():
    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\D\image'
    dest_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\D\label'
    display_functions.cut_via_color(img_path, dest_path)
    #copy_images(src_image_list, src_path, dest_path)




def main():
    #get_stems()
    #get_average_color()
    #use_get_average_color()
    get_stem_scores()
    #get_mean_hue()

if __name__ == '__main__':
    main()