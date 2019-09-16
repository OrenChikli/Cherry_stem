from work.unet.clarifruit_unet import unet_model, keras_functions
from work.preprocess import data_functions
from keras.callbacks import ReduceLROnPlateau
import cv2

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

logger.debug("123")


def splitter():
    raw_src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines\Source'

    split_dest = r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines\Data split'

    x_folder_name = 'image'
    y_folder_name = 'label'

    train_folder = 'train'
    test_folder = 'test'

    src_path = data_functions.image_train_test_split(raw_src_path, split_dest, x_folder_name, y_folder_name, test_size=0.3,
                                                     train_name=train_folder, test_name=test_folder)



def get_data_via_with_mask():
    src_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes'
    data_path = r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines'
    src_folder = 'with_maskes'
    x_folder = 'image'
    y_folder = 'label'

    data_functions.get_from(src_path, data_path, src_folder, x_folder, y_folder)

def train_unet():
    logger.debug(" <-train_unet")

    src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines\Data split\test_split_0.3\train'

    x_folder_name = 'image'
    y_folder_name = 'label'


    model_save_dest_path = r'D:\Clarifruit\cherry_stem\data\unet_data\model data'

    modes_dict = {'grayscale': 1, 'rgb': 3}  # translate for image dimentions

    target_size = (256, 256)
    color_mode = 'grayscale'

    weights_file_name = 'unet_cherry_stem.hdf5'

    data_gen_args = dict(rescale=1. / 255,
                         rotation_range=0.5,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='nearest')

    fit_params = dict(batch_size=10,
                      epochs=5,
                      steps_per_epoch=10,
                      validation_steps=10)

    params_dict = dict(src_path=src_path,
                       dest_path=model_save_dest_path,

                       x_folder_name=x_folder_name,
                       y_folder_name=y_folder_name,

                       target_size=target_size,
                       color_mode=color_mode,
                       input_size=(*target_size, modes_dict[color_mode]),

                       weights_file_name=weights_file_name,

                       data_gen_params=data_gen_args,
                       model_fit_params=fit_params)
    # early_stoping = EarlyStopping(monitor='val_loss',verbose=1, patience=3)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=2, min_lr=0.000001,
                                  cooldown=1, verbose=1)
    # callbacks = [early_stoping, model_checkpoint,reduce_lr]
    callbacks = [reduce_lr]

    model = keras_functions.clarifruit_train(params_dict, callbacks)

def use_predict():
    pre_trained_weights = r'D:\Clarifruit\cherry_stem\data\unet_data\model data\2019-09-15_17-23-27\unet_cherry_stem.hdf5'
    model = unet_model.unet(input_size=(256,256,1), pretrained_weights= pre_trained_weights)

    pred_path = r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines\Data split\test_split_0.3\pred\1'
    image_train_path = r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines\Data split\test_split_0.3\train\image'
    keras_functions.prediction(model, image_train_path, pred_path, (256, 256),
                               threshold=0.5, color_mode=cv2.IMREAD_GRAYSCALE)

if __name__ == '__main__':

    #train_unet()
    use_predict()