import os
import random
import shutil

def create_path(src_path, path_extention):
    new_path = os.path.join(src_path, path_extention)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path


def image_train_test_split(src_path, dest_path, x_name, y_name, test_size=0.3):
    train_path = create_path(dest_path, 'train')
    test_path = create_path(dest_path, 'test')

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

    return train_path, test_path


def copy_images(src_image_list, src_path, dest_path):
    for image in src_image_list:
        image_path = os.path.join(src_path, image)
        _ = shutil.copy(image_path, dest_path)


def create_X_y_paths(src_path, X_name, y_name):
    X_path = create_path(src_path, X_name)
    y_path = create_path(src_path, y_name)
    return X_path, y_path