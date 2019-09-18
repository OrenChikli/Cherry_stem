from keras.preprocessing.image import ImageDataGenerator

from work.preprocess import display_functions
import numpy as np
#import skimage.io as io
import skimage.transform as trans
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from work.preprocess.data_functions import *
from work.unet.clarifruit_unet.unet_model import *
#from tqdm import tqdm # this causes problems with kers progress bar in jupyter!!!
import json

import logging

logger = logging.getLogger(__name__)


# DATA GENERATORS


# batch_size=10, batch_size=10, epochs=5,steps_per_epoch=10, validation_steps=10

MODES_DICT = {'grayscale': 1, 'rgb': 3}  # translate for image dimentions

COLOR_TO_OPENCV = {'grayscale': 0, 'rgb': 1}
OPTIMIZER_DICT = {'Adam':Adam, 'adagrad':adagrad}


class ClarifruitUnet:

    def __init__(self, train_path, dest_path, weights_file_name,
                 x_folder_name='image', y_folder_name='label', test_path=None,
                 data_gen_args=None,
                 optimizer=None, optimizer_params=None, loss=None, metrics=None, pretrained_weights=None,
                 target_size=(256, 256), color_mode='rgb', mask_color_mode='grayscale',
                 batch_size=10, epochs=5, steps_per_epoch=10,
                 valdiation_split=0.2, validation_steps=10):

        self.train_path = train_path
        self.test_path = test_path
        self.dest_path = dest_path
        self.x_folder_name = x_folder_name
        self.y_folder_name = y_folder_name
        self.weights_file_name = weights_file_name

        self.weights_file_name = weights_file_name

        self.data_gen_args = data_gen_args

        self.target_size = target_size
        self.color_mode = color_mode
        self.input_size = (*target_size, MODES_DICT[color_mode])

        self.mask_color_mode = mask_color_mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        self.model = None
        self.optimizer = OPTIMIZER_DICT[optimizer](**optimizer_params)
        self.loss = loss
        self.metrics = metrics
        self.pretrained_weights = pretrained_weights

        self.train_generator = None
        self.val_generator = None
        self.model_checkpoint = None
        self.callbacks = None
        self.train_time = None

        self.save_to_dir = None
        self.validation_split = valdiation_split
        self.seed = 1



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

    def clarifruit_train_val_generators(self):


        image_train_generator, image_val_generator = self.train_val_generators(batch_size=self.batch_size,
                                                                               src_path=self.train_path,
                                                                               folder=self.x_folder_name,
                                                                               aug_dict=self.data_gen_args,
                                                                               color_mode=self.color_mode,
                                                                               save_prefix=self.x_folder_name,
                                                                               save_to_dir=self.save_to_dir,
                                                                               target_size=self.target_size,
                                                                               seed=self.seed,
                                                                               validation_split=self.validation_split)

        mask_train_generator, mask_val_generator = self.train_val_generators(batch_size=self.batch_size,
                                                                             src_path=self.train_path,
                                                                             folder=self.y_folder_name,
                                                                             aug_dict=self.data_gen_args,
                                                                             color_mode=self.mask_color_mode,
                                                                             save_prefix=self.y_folder_name,
                                                                             save_to_dir=self.save_to_dir,
                                                                             target_size=self.target_size,
                                                                             seed=self.seed,
                                                                             validation_split=self.validation_split)

        self.train_generator = zip(image_train_generator, mask_train_generator)
        self.val_generator = zip(image_val_generator, mask_val_generator)

        # return train_generator, val_generator

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

    def test_generator(self, test_path):
        logger.debug(" <-test_generator")
        logger.info(f"test generator params:test_path={test_path}\n"
                    f"target_size={self.target_size}\n,color_mode={self.color_mode}")

        img_list = os.scandir(test_path)
        for img_entry in img_list:

            img = cv2.imread(img_entry.path,COLOR_TO_OPENCV[self.color_mode])
            orig_shape = img.shape[:2]
            if self.color_mode == "grayscale":
                img = np.reshape(img, img.shape + (1,))
            img = img / 255
            img = trans.resize(img, self.target_size)
            img = np.reshape(img, (1,) + img.shape)
            yield img, img_entry, orig_shape

    def prediction(self, threshold=0.5):

        save_path = os.path.join(self.dest_path,self.train_time)
        save_path = create_path(save_path, 'pred')

        test_gen = self.test_generator(self.test_path)
        for img, img_entry,orig_shape in test_gen:
            img_name = img_entry.name.rsplit('.', 1)[0]
            img_name = img_name.rsplit('.', 1)[0]

            pred = self.model.predict(img, batch_size=1)

            save_img = cv2.imread(img_entry.path,COLOR_TO_OPENCV[self.color_mode])
            pred = cv2.resize(pred,orig_shape)

            pred_image_raw = (255 * pred[0]).astype(np.uint8)
            pred_img = (255 * (pred[0] > threshold)).astype(np.uint8)

            display_functions.overlay(save_img,pred_img,1)
            #save_img = (255 * img[0]).astype(np.uint8)

            cv2.imwrite(os.path.join(save_path, f"{img_name}.jpg"), save_img)
            cv2.imwrite(os.path.join(save_path, f"{img_name}_raw_predict.jpg"), pred_image_raw)
            cv2.imwrite(os.path.join(save_path, f"{img_name}_predict.png"), pred_img)

    def display_res_generator(self,save_path):
        pass

    def get_unet_model(self):
        self.model = unet(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics,
                          pretrained_weights=self.pretrained_weights,
                          input_size=self.input_size)

    def fit_unet(self):
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.val_generator,
            validation_steps=self.validation_steps,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=1)

        # def set_params(self, train_path=None, dest_path=None, data_gen_args=None,
        #                x_folder_name=None, y_folder_name=None, test_path=None,
        #                batch_size=None, epochs=None, steps_per_epoch=None, validation_steps=None, callbacks=None,
        #                target_size=None, color_mode=None, mask_color_mode=None,
        #                optimizer=None, loss=None, metrics=None, train_time=None):

    def set_params(self, **kwargs):

        if 'train_path' in kwargs:
            self.train_path = kwargs['train_path']

        if 'test_path' in kwargs:
            self.test_path = kwargs['test_path']

        if 'dest_path' in kwargs:
            self.dest_path = kwargs['dest_path']

        if 'x_folder_name' in kwargs:
            self.x_folder_name = kwargs['x_folder_name']

        if 'y_folder_name' in kwargs:
            self.y_folder_name = kwargs['y_folder_name']

        if 'data_gen_args' in kwargs:
            self.data_gen_args = kwargs['data_gen_args']

        if 'target_size' in kwargs:
            self.target_size = kwargs['target_size']
            self.input_size = (*self.target_size, MODES_DICT[self.color_mode])

        if 'color_mode' in kwargs:
            self.color_mode = kwargs['color_mode']
            self.input_size = (*self.target_size, MODES_DICT[self.color_mode])

        if 'mask_color_mode' in kwargs:
            self.mask_color_mode = kwargs['mask_color_mode']

        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']

        if 'epochs' in kwargs:
            self.epochs = kwargs['epochs']

        if 'steps_per_epoch' in kwargs:
            self.steps_per_epoch = kwargs['steps_per_epoch']

        if 'validation_steps' in kwargs:
            self.validation_steps = kwargs['validation_steps']

        if 'optimizer' in kwargs:
            self.optimizer = kwargs['optimizer']

        if 'loss' in kwargs:
            self.loss = kwargs['loss']

        if 'metrics' in kwargs:
            self.metrics = kwargs['metrics']

        if 'callbacks' in kwargs:
            self.callbacks = kwargs['callbacks']

        if 'train_time' in kwargs:
            self.train_time = kwargs['train_time']



    """
    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

    # later...

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    """

    def save_model(self, path_params, data_gen_args, unet_params, fit_params, optimizer_params):

        curr_folder = create_path(self.dest_path, self.train_time)
        self.model.save_weights(os.path.join(curr_folder,"model.hdf5"))
        save_settings_to_file(path_params, "path_params.json", curr_folder)
        save_settings_to_file(data_gen_args, "data_gen_args.json", curr_folder)
        save_settings_to_file(unet_params, "unet_params.json", curr_folder)
        save_settings_to_file(fit_params, "fit_params.json", curr_folder)
        save_settings_to_file(optimizer_params, "optimizer_params.json", curr_folder)





    def train_model(self,path_params, data_gen_args, unet_params, fit_params, callbacks=None,save_flag=True):


        self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if save_flag:
            curr_folder = create_path(self.dest_path, self.train_time)

            out_model_path = os.path.join(curr_folder, self.weights_file_name)
            model_checkpoint = [ModelCheckpoint(out_model_path, monitor='loss',
                                                verbose=1, save_best_only=True)]
            callbacks = model_checkpoint + callbacks
            self.callbacks = callbacks
            self.save_model(path_params, data_gen_args, unet_params, fit_params)
        else:
            self.callbacks = callbacks

        self.clarifruit_train_val_generators()
        self.get_unet_model()
        self.fit_unet()

