from __future__ import print_function

import os

import random
import shutil
import cv2
import matplotlib.pyplot as plt
import json
from work.auxiliary import decorators
from scipy.spatial.distance import euclidean
import pickle

import logging

logger = logging.getLogger(__name__)
logger_decorator = decorators.Logger_decorator(logger)


@logger_decorator.debug_dec
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

@logger_decorator.debug_dec
def get_masks_via_img(src_img_path,src_mask_path,dest_mask_path):
    """
    use the src_img_path with selected images to move respective masks from
     src_mask path to dest_mask_path
    :param src_img_path: folder containing the source images
    :param src_mask_path: folder containing all the predictions
    :param dest_mask_path: destination to move respective masks
    :return: None
    """
    for img_entry in os.scandir(src_img_path):
        npy_name = img_entry.name.rsplit(".",1)[0]+'.npy'
        curr_npy_path = os.path.join(src_mask_path,npy_name)
        curr_img_path = os.path.join(src_mask_path,img_entry.name)
        _=shutil.copy(curr_img_path,dest_mask_path)
        _=shutil.copy(curr_npy_path,dest_mask_path)


@logger_decorator.debug_dec
def get_from(src_path,data_path, src_folder, x_folder, y_folder):


    x_src_path = os.path.join(src_path, x_folder)
    y_src_path = os.path.join(src_path, y_folder)
    src_path = os.path.join(data_path, src_folder)

    images = [img.name for img in os.scandir(src_path)]


    x_dest_path = create_path(data_path, x_folder)
    y_dest_path = create_path(data_path, y_folder)

    copy_images(images,x_src_path,x_dest_path)
    copy_images(images,y_src_path,y_dest_path)


@logger_decorator.debug_dec
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


@logger_decorator.debug_dec
def save_json(params_dict, file_name, save_path):
    """
    save a given dict object as a json file
    :param params_dict: input dict object to be saved
    :param file_name: the save file name(or full path), must have .json file
     extention, e.g "rsults.json"
    :param save_path: the destination where to save the json file
    :return: None
    """

    dest = os.path.join(save_path, file_name)

    with open(dest, 'w') as f:
        json.dump(params_dict, f,indent=1)

@logger_decorator.debug_dec
def load_json(path):
    """
    loads a json file from the path
    :param path: the full path to the json file
    :return: a json.load object
    """

    with open(path, 'r') as f:
        res = json.load(f)
    return res





@logger_decorator.debug_dec
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



@logger_decorator.debug_dec
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

@logger_decorator.debug_dec
def get_train_test_split(src_path,dest_path,train_name='train',test_name='test',test_size=0.33):
    train_path=create_path(dest_path,train_name)
    test_path=create_path(dest_path,test_name)
    for folder_entry in os.scandir(src_path):
        curr_train_path = create_path(train_path,folder_entry.name)
        curr_test_path = create_path(test_path,folder_entry.name)
        n=len(os.listdir(folder_entry.path))
        ind = int((1-test_size) * n)
        train_inds = sorted(random.sample(range(n),ind))
        j=0
        for i,img_entry in enumerate(os.scandir(folder_entry.path)):
            if j<ind and i==train_inds[j]:
                _=shutil.copy(img_entry.path,curr_train_path)
                j+=1
            else:
                _=shutil.copy(img_entry.path,curr_test_path)
