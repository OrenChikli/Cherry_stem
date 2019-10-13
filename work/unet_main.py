from work.unet import unet_model_functions
from auxiliary import data_functions
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from logger_settings import *
import numpy as np

configure_logger()
logger = logging.getLogger("unet_main")


def train_model(train_path, params_dict, dest_path=None):
    """
    a method to train a model of unet on data from train_path. if dest path is provided, save the model to dest_path.
    withing the method one can specify various trainig variables and callbacks.
    :param train_path: location of training data, must have 2 sub folders, 1 for the "X" data, default is "image"
    and another for the masks or labels, default is "label
    :param dest_path: optional, path to which the trined model can be saved to
    :return: an instance of a trained model
    """
    logger.info(f"setting parameters to train on data from {train_path}")

    logger.info("created training instance")
    model = unet_model_functions.ClarifruitUnet(**params_dict)
    logger.info("train start")
    model.train_model(dest_path=dest_path, params_dict=params_dict)
    logger.info("finished training")
    return model


def load_from_files(src_path, updat_dict=None):
    """
    a method that can load a trained instance of ClarifruitUnet from a path in src_path variable
    the model parameters can be modified for further training as shown within
    :param src_path: the path where the trained model resides in
    :return: the ClarifruitUnet instance
    """
    params_dict = unet_model_functions.ClarifruitUnet.load_model(src_path)
    if updat_dict is not None:
        params_dict.update(updat_dict)
    logger.info(f"loading model from {src_path}")
    model = unet_model_functions.ClarifruitUnet(**params_dict)
    return model, params_dict


def main():
    train_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes'
    #train_path = r'D:\Clarifruit\unet\data\membrane\train'
    dest_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training'
    test_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    #test_path =r'D:\Clarifruit\unet\data\membrane\test'
    src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-10-13_14-57-43'

    params_dict = dict(

        train_path=train_path,
        x_folder_name='image',
        y_folder_name='label',
        weights_file_name='unet_cherry_stem.hdf5',

        data_gen_args=dict(rotation_range=180,
                           brightness_range=[0.2, 1.],
                           shear_range=10,
                           zoom_range=0.5,
                           horizontal_flip=True,
                           vertical_flip=True,
                           fill_mode='nearest'),
        seed=15,

        optimizer='Adam',
        optimizer_params=dict(lr=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy'],
        pretrained_weights=None,

        target_size=(256, 256),
        color_mode='grayscale',
        batch_size=10,
        epochs=10,
        steps_per_epoch=3000,
        valdiation_split=0.2,
        validation_steps=300)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=2, min_lr=0.000001,
                                  cooldown=1, verbose=1)

    early_stoping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=3,
                                  verbose=0,
                                  mode='auto',
                                  baseline=None,
                                  restore_best_weights=False)

    # callbacks = [reduce_lr, early_stoping]
    # params_dict['callbacks'] = callbacks

    # model = train_model(train_path=train_path,
    #                     params_dict=params_dict,
    #                     dest_path=dest_path)

    update_dict = dict(
        seed=10,
        update_freq=20,

        batch_size=10,
        epochs=5,
        steps_per_epoch=20,
        valdiation_split=0.2,
        validation_steps=10)

    model, params_dict = load_from_files(src_path,update_dict)
    model.train_model(dest_path=dest_path, params_dict=params_dict)
    model.prediction(test_path, dest_path)


if __name__ == '__main__':
    main()
