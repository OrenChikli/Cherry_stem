from __future__ import print_function

import os

import random
import shutil
import cv2
import matplotlib.pyplot as plt
import json
from work.auxiliary import decorators
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import logging
import pickle

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
def get_masks_via_img(src_img_path, src_mask_path, dest_mask_path):
    """
    use the src_img_path with selected images to move respective masks from
     src_mask path to dest_mask_path
    :param src_img_path: folder containing the source images
    :param src_mask_path: folder containing all the predictions
    :param dest_mask_path: destination to move respective masks
    :return: None
    """
    for img_entry in os.scandir(src_img_path):
        npy_name = img_entry.name.rsplit(".", 1)[0] + '.npy'
        curr_npy_path = os.path.join(src_mask_path, npy_name)
        curr_img_path = os.path.join(src_mask_path, img_entry.name)
        _ = shutil.copy(curr_img_path, dest_mask_path)
        _ = shutil.copy(curr_npy_path, dest_mask_path)


def create_raw_test_train_ground_truth(ground_path, mask_path, src_path):

    create_raw_ground_truth(ground_path=ground_path,
                            mask_path=mask_path,
                            dest_path=src_path,
                            dest_folder='train')

    create_raw_ground_truth(ground_path=ground_path,
                            mask_path=mask_path,
                            dest_path=src_path,
                            dest_folder='test')


def create_raw_ground_truth(ground_path, mask_path, dest_path,
                            dest_folder='train'):
    """
    getting respective masks of images in diffrent classes.
    a method to get the predictions from src_mask_path, into the y_folder in
    the ground_path, where the x_folder has the source images, selected by the
    user
    :param ground_path: a path containing the ground truth, has a structure of
    x_folder with src images, y_folder with labels
    :param mask_path: the image where all the available predictions reside
    """
    curr_ground_path = os.path.join(ground_path,dest_folder)
    curr_dest_path = create_path(dest_path, dest_folder)
    for curr_class in os.scandir(curr_ground_path):
        curr_dest = create_path(curr_dest_path, curr_class.name)

        logger.info(f"getting ground truth for class {curr_class.name}")
        logger.info(f"copying ground truth from {curr_class.path}")
        logger.info(f"copying ground truth to {curr_dest}")

        get_masks_via_img(curr_class.path, mask_path, curr_dest)


@logger_decorator.debug_dec
def save_pickle(object,file_name,save_path):
    file_save_path = os.path.join(save_path,file_name)
    with open(file_save_path, 'wb') as f:
        pickle.dump(object, f)

@logger_decorator.debug_dec
def load_pickle(path):
    """
    loads a json file from the path
    :param path: the full path to the json file
    :return: a json.load object
    """
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res


@logger_decorator.debug_dec
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


@logger_decorator.debug_dec
def load_data(path, hist_type):
    logger.debug(f"loading train data from:{path}")
    ret_df = pd.DataFrame()

    for label_folder in os.scandir(path):
        hist_folder = os.path.join(label_folder.path, f'{hist_type}_histograms')
        curr_df = load_npy_data(hist_folder)
        curr_df['label'] = label_folder.name

        ret_df = pd.concat((ret_df, curr_df))

    ret_df['label'] = ret_df['label'].astype('category')
    ret_df = shuffle(ret_df)
    ret_df.reset_index(inplace=True, drop=True)

    return ret_df


@logger_decorator.debug_dec
def get_from(src_path, data_path, src_folder, x_folder, y_folder):
    x_src_path = os.path.join(src_path, x_folder)
    y_src_path = os.path.join(src_path, y_folder)
    src_path = os.path.join(data_path, src_folder)

    images = [img.name for img in os.scandir(src_path)]

    x_dest_path = create_path(data_path, x_folder)
    y_dest_path = create_path(data_path, y_folder)

    copy_images(images, x_src_path, x_dest_path)
    copy_images(images, y_src_path, y_dest_path)


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
        json.dump(params_dict, f, indent=1)


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
def find_closest(item, item_dict):
    """
    calculate ecuclidean distance from item (a numpy object) to labeled items in
    items dict and return closest label
    :param item:
    :param item_dict:
    :return:
    """
    distances = [(euclidean(item, curr_color), label) for label, curr_color in
                 item_dict.items()]
    max = sorted(distances)
    best = max[0][1]
    return best


@logger_decorator.debug_dec
def get_train_test_split(src_path, dest_path, train_name='train',
                         test_name='test', test_size=0.33):
    train_path = create_path(dest_path, train_name)
    test_path = create_path(dest_path, test_name)
    for folder_entry in os.scandir(src_path):
        curr_train_path = create_path(train_path, folder_entry.name)
        curr_test_path = create_path(test_path, folder_entry.name)
        n = len(os.listdir(folder_entry.path))
        ind = int((1 - test_size) * n)
        train_inds = sorted(random.sample(range(n), ind))
        j = 0
        for i, img_entry in enumerate(os.scandir(folder_entry.path)):
            if j < ind and i == train_inds[j]:
                _ = shutil.copy(img_entry.path, curr_train_path)
                j += 1
            else:
                _ = shutil.copy(img_entry.path, curr_test_path)
