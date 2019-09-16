
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import skimage.io as io
import skimage.transform as trans
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from work.preprocess.data_functions import *
from work.unet.clarifruit_unet.unet_model import *

import logging

logger = logging.getLogger(__name__)


# DATA GENERATORS


# batch_size=10, batch_size=10, epochs=5,steps_per_epoch=10, validation_steps=10

class ClarifruitUnet:

    def __init__(self,train_path, dest_path, weights_file_name, data_gen_args,
             x_folder_name='image', y_folder_name='label', test_path=None,
             batch_size=10, epochs=5,steps_per_epoch=10,validation_steps=10,
             target_size=(256,256), color_mode='rgb', mask_color_mode='grayscale'):

        self.train_path = train_path
        self.test_path = test_path
        self.dest_path = dest_path
        self.x_folder_name = x_folder_name
        self.y_folder_name = y_folder_name
        self.weights_file_name = weights_file_name

        self.weights_file_name = weights_file_name

        self.data_gen_args = data_gen_args

        self.target_size = target_size
        self.modes_dict = {'grayscale': 1, 'rgb': 3}  # translate for image dimentions
        self.color_mode = color_mode
        self.input_size = (*target_size, self.modes_dict[color_mode])

        self.mask_color_mode = mask_color_mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        self.model = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.model_checkpoint = None
        self.callbacks = None


    @staticmethod
    def custom_generator(batch_size, src_path, folder, aug_dict, save_prefix,
                         color_mode="grayscale",
                         save_to_dir=None,
                         target_size=(256, 256),
                         seed=1):
        """
        Create a datagen generator
        :param batch_size: the batch size of each step
        :param src_path: the path to the data
        :param folder: the name of the folder in the data_path
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

    @staticmethod
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
        :param folder: the name of the folder in the data_path
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


    def clarifruit_train_val_generators(self,save_to_dir=None,validation_split=0.2,seed=1):
        """

        Creates train&val image generators with images and masks

        :param batch_size: the batch size of each step
        :param src_path: the path to the data
        :param image_folder: the name of the image folder in the data_path
        :param mask_folder: the name of the mask folder in the data_path
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

        image_train_generator, image_val_generator = self.train_val_generators(batch_size=self.batch_size,
                                                                          src_path=self.train_path,
                                                                          folder=self.x_folder_name,
                                                                          aug_dict=self.data_gen_args,
                                                                          color_mode=self.color_mode,
                                                                          save_prefix=self.x_folder_name,
                                                                          save_to_dir=save_to_dir,
                                                                          target_size=self.target_size,
                                                                          seed=seed,
                                                                          validation_split=validation_split)

        mask_train_generator, mask_val_generator = self.train_val_generators(batch_size=self.batch_size,
                                                                        src_path=self.train_path,
                                                                        folder=self.y_folder_name,
                                                                        aug_dict=self.data_gen_args,
                                                                        color_mode=self.mask_color_mode,
                                                                        save_prefix=self.y_folder_name,
                                                                        save_to_dir=save_to_dir,
                                                                        target_size=self.target_size,
                                                                        seed=seed,
                                                                        validation_split=validation_split)

        self.train_generator = zip(image_train_generator, mask_train_generator)
        self.val_generator = zip(image_val_generator, mask_val_generator)

        #return train_generator, val_generator

    @staticmethod
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
        image_generator = ClarifruitUnet.custom_generator(batch_size=batch_size,
                                           src_path=train_path,
                                           folder=image_folder,
                                           aug_dict=aug_dict,
                                           save_prefix=image_save_prefix,
                                           color_mode=image_color_mode,
                                           save_to_dir=save_to_dir,
                                           target_size=target_size,
                                           seed=seed)

        mask_generator = ClarifruitUnet.custom_generator(batch_size=batch_size,
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


    def test_generator(self,test_path):
        logger.debug(" <-test_generator")
        logger.info(f"test generator params:test_path={test_path}\n"
                    f"target_size={self.target_size}\n,color_mode={self.color_mode}")

        img_list = os.scandir(test_path)
        for img_entry in img_list:

            img = cv2.imread(img_entry.path)
            if self.color_mode == "grayscale":
                img = np.reshape(img, img.shape +(1,))
            img = img / 255
            img = trans.resize(img, self.target_size)
            img = np.reshape(img, (1,) + img.shape)
            yield img, img_entry.name


    def prediction(self,test_path, threshold=0.5):

        logger.debug("<-prediction with params:\n"
                     f"test_path={test_path}\nsave_path={self.save_path}\ntarget_size={self.target_size}\n"
                     f"threshold={threshold}\ncolor_mode ={self.color_mode}")

        save_path = create_path(self.save_path,'pred')

        test_gen = self.test_generator(test_path)
        for img, img_name in test_gen:
            img_name = img_name.rsplit('.', 1)[0]
            img_name = img_name.rsplit('.', 1)[0]

            pred = self.model.predict(img, batch_size=1)

            pred_image_raw = (255 * pred[0]).astype(np.uint8)
            pred_img = (255 * (pred[0] > threshold)).astype(np.uint8)
            save_img = (255 * img[0]).astype(np.uint8)

            io.imsave(os.path.join(save_path, f"{img_name}.jpg"), save_img)
            io.imsave(os.path.join(save_path, f"{img_name}_raw_predict.jpg"), pred_image_raw)
            io.imsave(os.path.join(save_path, f"{img_name}_predict.png"), pred_img)


    def fit_unet(self):
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.val_generator,
            validation_steps=self.validation_steps,
            epochs=self.epochs,
            callbacks=self.callbacks)

    def set_params(self, train_path=None, dest_path=None, data_gen_args=None,
                 x_folder_name=None, y_folder_name=None, test_path=None,
                 batch_size=None, epochs=None, steps_per_epoch=None, validation_steps=None,callbacks=None,
                 target_size=None, color_mode=None, mask_color_mode=None):

        if train_path is not None:
            self.train_path = train_path
        if test_path is not None:
            self.test_path = test_path
        if dest_path is not None:
            self.dest_path = dest_path
        if x_folder_name is not None:
            self.x_folder_name = x_folder_name
        if y_folder_name is not None:
            self.y_folder_name = y_folder_name
        if data_gen_args is not None:
            self.data_gen_args = data_gen_args

        if target_size is not None:
            self.target_size = target_size
            self.input_size = (*target_size, self.modes_dict[color_mode])
        if color_mode is not None:
            self.color_mode = color_mode
            self.input_size = (*target_size, self.modes_dict[color_mode])
        if mask_color_mode is not None:
            self.mask_color_mode = mask_color_mode

        if batch_size is not None:
            self.batch_size = batch_size
        if epochs is not None:
            self.epochs = epochs
        if steps_per_epoch is not None:
            self.steps_per_epoch = steps_per_epoch
        if validation_steps is not None:
            self.validation_steps = validation_steps

        if self.batch_size is not None:
            self.batch_size = batch_size

        if self.steps_per_epoch is not None:
            self.steps_per_epoch = steps_per_epoch

        if self.validation_steps is not None:
            self.validation_steps = validation_steps

        if self.callbacks is not None:
            self.callbacks = callbacks

        """
        if model is not None:
            self.model = None
        if train_path is not None:
            self.train_generator = None
        if train_path is not None:
            self.val_generator = None
        if train_path is not None:
            self.test_generator = None
        if train_path is not None:
            self.model_checkpoint = None
        if train_path is not None:
            self.callbacks = None
        """

    def set_model_checkpoints(self,curr_time):
        weights_file_out = []
        self.model_checkpoint = [ModelCheckpoint(weights_file_out, monitor='loss', verbose=1, save_best_only=True)]







def init_model():
    init_params = dict(
        train_path=r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines\Data split\test_split_0.3\train',
        test_path=r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines\Data split\test_split_0.3\test',
        x_folder_name='image',
        y_folder_name='label',
        dest_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training',
        weights_file_name='unet_cherry_stem.hdf5')

    data_gen_args = dict(rescale=1. / 255,
                         rotation_range=0.5,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='nearest')

    fit_params = dict(target_size=(256, 256),
                      color_mode='grayscale',
                      batch_size=10,
                      epochs=5,
                      steps_per_epoch=10,
                      validation_steps=10)



    model = ClarifruitUnet(**init_params)
    train_model(model,fit_params,data_gen_args)


def train_model(model,fit_params,data_gen_args):
    curr_folder = create_path(save_path, current_time)
    model.set_params(**fit_params)
    model.set_params(data_gen_args=data_gen_args)
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')




    save_settings_to_file(params_dict, "model_settings.txt", curr_folder)






    cherry_model.clarifruit_train_val_generators(save_to_dir=None,validation_split=0.2,seed=1)

    weights_file_out = os.path.join(curr_folder, params_dict['weights_file_name'])
    model_checkpoint = [ModelCheckpoint(weights_file_out, monitor='loss', verbose=1, save_best_only=True)]
    callbacks = model_checkpoint + callbacks_in

    cherry_model.model = unet(input_size=params_dict['input_size'], pretrained_weights=pre_weights_file_path)



    return model,current_time