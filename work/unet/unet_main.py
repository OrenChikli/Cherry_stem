from work.unet.clarifruit_unet import  keras_functions
from work.preprocess import data_functions
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import *
import os


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('unet_main.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def splitter():
    raw_src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines\Source'

    split_dest = r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines\Data split'

    x_folder_name = 'image'
    y_folder_name = 'label'

    train_folder = 'train'
    test_folder = 'test'

    src_path = data_functions.image_train_test_split(raw_src_path, split_dest, x_folder_name, y_folder_name,
                                                     test_size=0.3,
                                                     train_name=train_folder, test_name=test_folder)


def get_data_via_with_mask():
    src_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes'
    data_path = r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines'
    src_folder = 'with_maskes'
    x_folder = 'image'
    y_folder = 'label'

    data_functions.get_from(src_path, data_path, src_folder, x_folder, y_folder)


def main():
    path_params = dict(
        train_path=r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes',
        test_path=r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig',
        x_folder_name='image',
        y_folder_name='label',
        dest_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training',
        weights_file_name='unet_cherry_stem.hdf5')

    data_gen_args = dict(rescale=1. / 255,
                         rotation_range=0.5,
                         width_shift_range=0.25,
                         height_shift_range=0.25,
                         shear_range=0.05,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='nearest')

    optimizer_params =dict(lr=1e-4)

    extra_params = dict(data_gen_args=data_gen_args,
                        optimizer_params=optimizer_params)

    unet_params = dict(optimizer='Adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'],
                       pretrained_weights=None)
    #r'D:\Clarifruit\cherry_stem\data\unet_data\model data\2019-09-15_15-33-49\unet_cherry_stem.hdf5')

    fit_params = dict(target_size=(256, 256),
                      color_mode='grayscale',
                      mask_color_mode='grayscale',
                      batch_size=10,
                      epochs=1,
                      steps_per_epoch=10,
                      valdiation_split=0.2,
                      validation_steps=10)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=2, min_lr=0.000001,
                                  cooldown=1, verbose=1)

    callbacks = [reduce_lr]

    init_dict = data_functions.join_dicts(path_params,unet_params,fit_params,extra_params)

    model = keras_functions.ClarifruitUnet(**init_dict)
    #model.clarifruit_train_val_generators()
    #model.get_unet_model()

    model.train_model(path_params, data_gen_args, unet_params,fit_params,optimizer_params,
                      callbacks=callbacks,saveflag=True)

    #model.prediction()

def load_from_files():
    src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-18_22-19-04'
    path_params, data_gen_args, unet_params, fit_params, optimizer_params = data_functions.load_model(src_path)

    extra_params = dict(data_gen_args=data_gen_args,
                        optimizer_params=optimizer_params)

    init_dict = data_functions.join_dicts(path_params, unet_params, fit_params, extra_params)

    model = keras_functions.ClarifruitUnet(**init_dict)
    model.set_params(train_time=os.path.basename(src_path))
    model.clarifruit_train_val_generators()
    model.get_unet_model()
    model.prediction()
    #model.train_model(path_params, data_gen_args, unet_params,fit_params,save_flag=False)
    #model.save_model(path_params, data_gen_args, unet_params,fit_params,optimizer_params)

if __name__ == '__main__':
    #main()

    load_from_files()