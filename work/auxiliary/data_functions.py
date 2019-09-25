from __future__ import print_function

import os

import random
import shutil
import cv2
import matplotlib.pyplot as plt
import json
from work.auxiliary import display_functions
from scipy.spatial.distance import euclidean



# PATH FUNCTIONS


def create_path(src_path, path_extention):
    """
    create a new folder called the path_extention value in the src_path
    :param src_path: the mother path to create a new folder
    :param path_extention: new folder to create
    :return: the path of the created folder
    """
    new_path = os.path.join(src_path, path_extention)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path


def get_masks_via_img(src_img_path,src_mask_path,dest_mask_path):
    for img_entry in os.scandir(src_img_path):
        curr_src_path = os.path.join(src_mask_path,img_entry.name)
        _=shutil.copy(curr_src_path,dest_mask_path)



def get_from(src_path,data_path, src_folder, x_folder, y_folder):


    x_src_path = os.path.join(src_path, x_folder)
    y_src_path = os.path.join(src_path, y_folder)
    src_path = os.path.join(data_path, src_folder)

    images = [img.name for img in os.scandir(src_path)]


    x_dest_path = create_path(data_path, x_folder)
    y_dest_path = create_path(data_path, y_folder)

    copy_images(images,x_src_path,x_dest_path)
    copy_images(images,y_src_path,y_dest_path)



def copy_images(src_image_list, src_path, dest_path):
    """
    copies a list of src images in src_path to a new destination in dest path
    :param src_image_list: a list of image names in src_path
    :param src_path:
    :param dest_path:
    :return:
    """
    for image in src_image_list:
        image_path = os.path.join(src_path, image)
        _ = shutil.copy(image_path, dest_path)



def save_json(params_dict, file_name, save_path):
    """
    save a given dict object as a json file
    :param params_dict: input dict object to be saved
    :param file_name: the save file name(or full path), must have .json file extention, e.g "rsults.json"
    :param save_path: the destination where to save the json file
    :return: None
    """

    dest = os.path.join(save_path, file_name)

    with open(dest, 'w') as f:
        json.dump(params_dict, f,indent=1)

def load_json(path):
    """
    loads a json file from the path
    :param path: the full path to the json file
    :return: a json.load object
    """

    with open(path, 'r') as f:
        res = json.load(f)
    return res



def plot_res(test_img, ground_truth_mask, test_mask_raw):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax = ax.flatten()

    ax[0].set_title('image')
    ax[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

    ax[2].set_title('ground truth')
    ax[2].imshow(ground_truth_mask)

    ax[3].set_title('raw pred mask')
    ax[3].imshow(test_mask_raw)

    plt.show()




def get_with_maskes(img_path,mask_path,dest_path,color=(0,255,255)):
    img_list = os.scandir(img_path)
    for img_entry in img_list:
        img = cv2.imread(img_entry.path,cv2.IMREAD_UNCHANGED)
        curr_mask_path = os.path.join(mask_path,img_entry.name)
        mask = cv2.imread(curr_mask_path,cv2.IMREAD_UNCHANGED)
        ontop = display_functions.put_binary_ontop(img,mask,color)
        curr_dest_path = os.path.join(dest_path,img_entry.name)
        cv2.imwrite(curr_dest_path,ontop)


def find_closest(item,item_dict):
    """
    calculate ecuclidean distance from item (a numpy object) to labeled items in
    items dict and return closest label
    :param item:
    :param item_dict:
    :return:
    """
    distances = [(euclidean(item, curr_color), label) for label,curr_color in item_dict.items()]
    max = sorted(distances)
    best = max[0][1]
    return best