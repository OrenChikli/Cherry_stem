from __future__ import print_function

import os

import random
import shutil
import cv2
import matplotlib.pyplot as plt
import json
from work.preprocess import display_functions

from pathlib import Path



# PATH FUNCTIONS


def create_path(src_path, path_extention):
    new_path = os.path.join(src_path, path_extention)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path


def image_train_test_split(src_path, dest_path, x_name, y_name, test_size=0.3,test_name='test',train_name='train'):
    cur_test_path = create_path(dest_path,f'test_split_{test_size}')
    train_path = create_path(cur_test_path, train_name)
    test_path = create_path(cur_test_path, test_name)

    X_path = os.path.join(src_path, x_name)
    y_path = os.path.join(src_path, y_name)

    X_train_path, y_train_path = create_X_y_paths(train_path, x_name, y_name)
    X_test_path, y_test_path = create_X_y_paths(test_path, x_name, y_name)

    # get images list in src folder
    img_list = [f for f in os.listdir(X_path)]

    random.shuffle(img_list)
    split_ind = int(test_size * len(img_list))

    train_data = img_list[split_ind:]
    copy_images(train_data, X_path, X_train_path)
    copy_images(train_data, y_path, y_train_path)

    test_data = img_list[:split_ind]
    copy_images(test_data, X_path, X_test_path)
    copy_images(test_data, y_path, y_test_path)

    return cur_test_path


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
    for image in src_image_list:
        image_path = os.path.join(src_path, image)
        _ = shutil.copy(image_path, dest_path)


def create_X_y_paths(src_path, X_name, y_name):
    X_path = create_path(src_path, X_name)
    y_path = create_path(src_path, y_name)
    return X_path, y_path


def save_settings_to_file(params_dict, file_name, save_path):

    dest = os.path.join(save_path, file_name)

    with open(dest, 'w') as f:
        json.dump(params_dict, f,indent=1)


# IMAGE FUNCTIONS

def load_resize_img(img_folder, image_name, target_size):
    img_path = os.path.join(img_folder, image_name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, target_size)
    return img


def get_images(image_name, pred_path, test_image_path, test_mask_path, target_size):
    test_image_name = image_name + '.jpg'
    test_img = load_resize_img(test_image_path, test_image_name, target_size)

    ground_truth_name = image_name + '.jpg'
    ground_truth_mask = load_resize_img(test_mask_path, ground_truth_name, target_size)

    test_mask_raw_name = image_name + '_raw_predict.png'
    test_mask_raw = load_resize_img(pred_path, test_mask_raw_name, target_size)

    test_mask_binary_name = image_name + '_predict.png'
    test_mask_binary = load_resize_img(pred_path, test_mask_binary_name, target_size)

    return test_img, ground_truth_mask, test_mask_raw, test_mask_binary


def plot_from_list(file_list,pred_path,orig_image_path,orig_mask_path,target_size):
    for file_name in file_list:
        test_img, ground_truth_mask, test_mask_raw, _ = get_images(file_name,
                                                                          pred_path,
                                                                          orig_image_path,
                                                                          orig_mask_path,
                                                                          target_size)
        print(f"Results for image :{file_name}")
        plot_res(test_img, ground_truth_mask, test_mask_raw)





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

def load_model(src_path):
    params_dict = {}
    pretrained_weights={}
    files = os.scandir(src_path)
    for file_entry in files:
        file_name_segments = file_entry.name.rsplit('.', 1)
        file_name = file_name_segments[0]
        file_extention = file_name_segments[-1]
        if file_extention == 'json':
            with open(file_entry.path, 'r') as f:
                params_dict = json.load(f)

        elif file_extention == 'hdf5':
            pretrained_weights = file_entry.path

    params_dict['pretrained_weights'] = pretrained_weights
    params_dict['train_time'] = os.path.basename(src_path)

    return params_dict


def get_with_maskes(img_path,mask_path,dest_path,color=(0,255,255)):
    img_list = os.scandir(img_path)
    for img_entry in img_list:
        img = cv2.imread(img_entry.path,cv2.IMREAD_UNCHANGED)
        curr_mask_path = os.path.join(mask_path,img_entry.name)
        mask = cv2.imread(curr_mask_path,cv2.IMREAD_UNCHANGED)
        ontop = display_functions.put_binary_ontop(img,mask,color)
        curr_dest_path = os.path.join(dest_path,img_entry.name)
        cv2.imwrite(curr_dest_path,ontop)