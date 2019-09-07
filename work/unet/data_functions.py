from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
import skimage.io as io
import skimage.transform as trans



def custom_generator(batch_size, src_path, folder, aug_dict, save_prefix,
                     color_mode="grayscale",
                     save_to_dir=None,
                     target_size=(256, 256),
                     seed=1):
    """
    Create a datagen generator
    :param batch_size: the batch size of each step
    :param src_path: the path to the data
    :param folder: the name of the folder in the src_path
    :param aug_dict: a dictionary with the data augmentation parameters of the images
    :param save_prefix: if output images are saved, this is the prefix in the file names
    :param color_mode: how to load the images, options are "grayscale", "rgb", default is "grayscale"
    :param save_to_dir: path to save output images, if None nothing is saved, default is None
    :param target_size: pixel size of output images,default is (256,256)
    :param seed: random seed used in image generation, default is 1
    :return: a flow_from_dictionary keras datagenerator
    """
    datagen = ImageDataGenerator(**aug_dict)

    gen = datagen.flow_from_directory(
        src_path,
        classes=[folder],
        class_mode=None,
        color_mode=color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        seed=seed)

    return gen


def train_val_generators(batch_size, src_path, folder, aug_dict, save_prefix,
                         color_mode="grayscale",
                         save_to_dir=None,
                         target_size=(256, 256),
                         validation_split=0.2,
                         seed=1):
    """

    Create a datagen generator with  train validation split
    :param batch_size: the batch size of each step
    :param src_path: the path to the data
    :param folder: the name of the folder in the src_path
    :param aug_dict: a dictionary with the data augmentation parameters of the images
    :param save_prefix: if output images are saved, this is the prefix in the file names
    :param color_mode: how to load the images, options are "grayscale", "rgb", default is "grayscale"
    :param save_to_dir: path to save output images, if None nothing is saved, default is None
    :param target_size: pixel size of output images,default is (256,256)
    :param validation_split: size of the validation data set, default is 0.2
    :param seed: random seed used in image generation, default is 1
    :return: a flow_from_dictionary keras datagenerator
    """
    datagen = ImageDataGenerator(**aug_dict, validation_split=validation_split)

    train_gen = datagen.flow_from_directory(
        src_path,
        classes=[folder],
        class_mode=None,
        color_mode=color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        seed=seed,
        subset='training')

    val_gen = datagen.flow_from_directory(
        src_path,
        classes=[folder],
        class_mode=None,
        color_mode=color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        seed=seed,
        subset='validation')

    return train_gen, val_gen


def clarifruit_train_val_generators(batch_size, src_path, image_folder, mask_folder, aug_dict,
                                    image_color_mode="grayscale", mask_color_mode="grayscale",
                                    image_save_prefix="image", mask_save_prefix="mask",
                                    save_to_dir=None,
                                    target_size=(256, 256),
                                    validation_split=0.2,
                                    seed=1):
    """

    Creates train&val image generators with images and masks

    :param batch_size: the batch size of each step
    :param src_path: the path to the data
    :param image_folder: the name of the image folder in the src_path
    :param mask_folder: the name of the mask folder in the src_path
    :param aug_dict: a dictionary with the data augmentation parameters of the images
    :param image_color_mode: how to load the images, options are "grayscale", "rgb", default is "grayscale"
    :param mask_color_mode: how to load the masks, options are "grayscale", "rgb", default is "grayscale"
    :param image_save_prefix: if output images are saved, this is the prefix in the file names
    :param mask_save_prefix: if output masks are saved, this is the prefix in the file names
    :param save_to_dir: path to save output images, if None nothing is saved, default is None
    :param target_size: pixel size of output images,default is (256,256)
    :param validation_split: size of the validation data set, default is 0.2
    :param seed: random seed used in image generation, default is 1
    :return: a flow_from_dictionary keras datagenerator
    """

    image_train_generator, image_val_generator = train_val_generators(batch_size=batch_size,
                                                                      src_path=src_path,
                                                                      folder=image_folder,
                                                                      aug_dict=aug_dict,
                                                                      color_mode=image_color_mode,
                                                                      save_prefix=image_save_prefix,
                                                                      save_to_dir=save_to_dir,
                                                                      target_size=target_size,
                                                                      seed=seed,
                                                                      validation_split=validation_split)

    mask_train_generator, mask_val_generator = train_val_generators(batch_size=batch_size,
                                                                    src_path=src_path,
                                                                    folder=mask_folder,
                                                                    aug_dict=aug_dict,
                                                                    color_mode=mask_color_mode,
                                                                    save_prefix=mask_save_prefix,
                                                                    save_to_dir=save_to_dir,
                                                                    target_size=target_size,
                                                                    seed=seed,
                                                                    validation_split=validation_split)

    train_generator = zip(image_train_generator, mask_train_generator)
    val_generator = zip(image_val_generator, mask_val_generator)

    return train_generator, val_generator


def clarifruit_train_generator(batch_size, train_path, image_folder, mask_folder, aug_dict,
                               image_color_mode="grayscale", mask_color_mode="grayscale",
                               image_save_prefix="image",
                               mask_save_prefix="mask",
                               save_to_dir=None,
                               target_size=(256, 256),
                               seed=1):
    """
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    """

    image_generator = custom_generator(batch_size=batch_size,
                                       src_path=train_path,
                                       folder=image_folder,
                                       aug_dict=aug_dict,
                                       save_prefix=image_save_prefix,
                                       color_mode=image_color_mode,
                                       save_to_dir=save_to_dir,
                                       target_size=target_size,
                                       seed=seed)

    mask_generator = custom_generator(batch_size=batch_size,
                                      src_path=train_path,
                                      folder=mask_folder,
                                      aug_dict=aug_dict,
                                      save_prefix=mask_save_prefix,
                                      color_mode=mask_color_mode,
                                      save_to_dir=save_to_dir,
                                      target_size=target_size,
                                      seed=seed)

    train_generator = zip(image_generator, mask_generator)

    return train_generator


def test_generator(test_path, target_size=(256, 256), as_gray=True):
    img_list = os.scandir(test_path)
    for img_name in img_list:
        img = io.imread(os.path.join(test_path, img_name), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, (1,) + img.shape)
        yield img, img_name


def prediction(model,test_path, save_path, target_size,threshold=0.5, as_gray=False):
    test_gen = test_generator(test_path, target_size=target_size, as_gray=as_gray)
    for img, img_Entry in test_gen:
        img_name = img_Entry.name.rsplit('.', 1)[0]
        pred = model.predict(img, batch_size=1)
        pred_img = 255 * (pred[0] > threshold)
        pred_img = pred_img.astype(np.uint8)
        save_img = 255 * img[0]
        save_img = save_img.astype(np.uint8)
        io.imsave(os.path.join(save_path, f"{img_name}.png"), save_img)
        io.imsave(os.path.join(save_path, f"{img_name}_predict.png"), pred_img)
