from keras.preprocessing.image import ImageDataGenerator

from work.preprocess import display_functions
import numpy as np

import skimage.transform as trans
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from work.preprocess.data_functions import *
from keras.optimizers import *
from work.unet.clarifruit_unet.unet_model import unet
from work.preprocess import display_functions
from work.segmentation.clarifruit_segmentation import segmentation1
from work.unet.clarifruit_unet import unet_model
from work.segmentation.clarifruit_segmentation import *
#from tqdm import tqdm # this causes problems with kers progress bar in jupyter!!!
import json
from keras.models import model_from_json


import logging

logger = logging.getLogger(__name__)


# DATA GENERATORS


MODES_DICT = {'grayscale': 1, 'rgb': 3}  # translate for image dimentions

COLOR_TO_OPENCV = {'grayscale': 0, 'rgb': 1}
OPTIMIZER_DICT = {'Adam':Adam, 'adagrad':adagrad}


class ClarifruitUnet:

    def __init__(self, train_path, dest_path, weights_file_name,
                 x_folder_name='image', y_folder_name='label', test_path=None,
                 data_gen_args=None,callbacks=None,
                 optimizer=None, optimizer_params=None, loss=None, metrics=None, pretrained_weights=None,
                 target_size=(256, 256), color_mode='rgb', mask_color_mode='grayscale',
                 batch_size=10, epochs=5, steps_per_epoch=10,
                 valdiation_split=0.2, validation_steps=10,
                 train_time=None):

        logger.debug(" <- __init__")

        self.train_path = train_path
        self.test_path = test_path
        self.dest_path = dest_path
        self.x_folder_name = x_folder_name
        self.y_folder_name = y_folder_name
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
        self.callbacks = callbacks
        self.train_time = train_time

        self.save_to_dir = None
        self.validation_split = valdiation_split
        self.seed = 1

        self.clarifruit_train_val_generators()
        #self.get_generators()
        self.get_unet_model()

        logger.debug(" -> __init__")

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
        logger.debug(f" <- custom_generator, src:\n{src_path}")
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
        logger.debug(f"-> custom_generator, src:\n{src_path}")
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
        logger.debug(f" <- train_val_generators src:\n{src_path}")
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
        logger.debug(f" -> train_val_generators src:\n{src_path}")
        return train_gen, val_gen



    def from_path_generators(self,path):
        logger.debug(f" <- clarifruit_train_val_generators")
        image_train_generator, image_val_generator = self.train_val_generators(batch_size=self.batch_size,
                                                                               src_path=path,
                                                                               folder=self.x_folder_name,
                                                                               aug_dict=self.data_gen_args,
                                                                               color_mode=self.color_mode,
                                                                               save_prefix=self.x_folder_name,
                                                                               save_to_dir=self.save_to_dir,
                                                                               target_size=self.target_size,
                                                                               seed=self.seed,
                                                                               validation_split=self.validation_split)

        mask_train_generator, mask_val_generator = self.train_val_generators(batch_size=self.batch_size,
                                                                             src_path=path,
                                                                             folder=self.y_folder_name,
                                                                             aug_dict=self.data_gen_args,
                                                                             color_mode=self.mask_color_mode,
                                                                             save_prefix=self.y_folder_name,
                                                                             save_to_dir=self.save_to_dir,
                                                                             target_size=self.target_size,
                                                                             seed=self.seed,
                                                                             validation_split=self.validation_split)

        train_generator = zip(image_train_generator, mask_train_generator)
        val_generator = zip(image_val_generator, mask_val_generator)
        logger.debug(f" -> clarifruit_train_val_generators")
        return train_generator,val_generator

    def get_generators(self):
        white_path = r'D:\Clarifruit\cherry_stem\data\raw_data\class_seperated\white'
        blank_path = r'D:\Clarifruit\cherry_stem\data\raw_data\class_seperated\blank'
        red_black_path = r'D:\Clarifruit\cherry_stem\data\raw_data\class_seperated\red_black'

        paths = [red_black_path,white_path,blank_path]
        train_generators = []
        val_generators = []
        for path in paths:
            train,val = self.from_path_generators(path)
            train_generators += train
            val_generators += val

        self.val_generator = val_generators
        self.train_generator = train_generators

    def clarifruit_train_val_generators(self):
        logger.debug(f" <- clarifruit_train_val_generators")
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
        logger.debug(f" -> clarifruit_train_val_generators")



    def test_generator(self, test_path):
        logger.debug(" <-test_generator")

        img_list = os.scandir(test_path)
        for img_entry in img_list:

            img = cv2.imread(img_entry.path,COLOR_TO_OPENCV[self.color_mode])
            if img.shape[-1] == 3:
                orig_shape = img.shape[-2::-1]
            else:
                orig_shape = img.shape[::-1]
            if self.color_mode == "grayscale":
                img = np.reshape(img, img.shape + (1,))
            img = img / 255
            img = trans.resize(img, self.target_size)
            img = np.reshape(img, (1,) + img.shape)
            yield img, img_entry, orig_shape


    def prediction(self,threshold=0.5):
        logger.debug(" <- prediction")
        if self.train_time is None:
            self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        save_path = create_path(self.dest_path,self.train_time)
        #threshold_path = create_path(save_path, f'binary_thres_{threshold}')
        #mask_on_path = create_path(save_path,'mask_ontop')
        save_path = create_path(save_path, 'raw_pred')


        test_gen = self.test_generator(self.test_path)
        for img, img_entry,orig_shape in test_gen:

            pred = self.model.predict(img, batch_size=1)[0]

            pred_image_raw = (255 * pred).astype(np.uint8)
            pred_image_raw = cv2.resize(pred_image_raw, orig_shape)
            cv2.imwrite(os.path.join(save_path, img_entry.name), pred_image_raw)

            #pred_image_thres = (255 * (pred > threshold)).astype(np.uint8)
            #pred_image_thres = cv2.resize(pred_image_thres, orig_shape)
            #cv2.imwrite(os.path.join(threshold_path, img_entry.name), pred_image_thres)

            #real_img = cv2.imread(img_entry.path,cv2.IMREAD_UNCHANGED)
            #thres_ontop = display_functions.put_binary_ontop(real_img,pred_image_thres)
            #cv2.imwrite(os.path.join(mask_on_path, img_entry.name), thres_ontop)

        logger.debug(" -> prediction")




    def get_unet_model(self):
        logger.debug(" <- get_unet_model")
        self.model = unet(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics,
                          pretrained_weights=self.pretrained_weights,
                          input_size=self.input_size)
        logger.debug(" -> get_unet_model")


    def fit_unet(self):
        logger.debug(" <- fit_unet")

        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.val_generator,
            validation_steps=self.validation_steps,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=1)
        logger.debug(" -> fit_unet")

    def set_params(self, **kwargs):
        logger.debug(" <- set_params")

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

        logger.debug(" -> set_params")


    def save_model(self, params_dict,checkpoint_flag=False):
        logger.debug(" <- save_model")
        if checkpoint_flag:
            curr_folder = self.set_model_checkpint()
        else:
            curr_folder = self.get_curr_folder()

        save_dict = params_dict.copy()
        if 'callbacks' in save_dict:
            save_dict.pop('callbacks')
        save_json(save_dict, "model_params.json", curr_folder)
        #model_json = self.model.to_json()

        #with open(os.path.join(curr_folder,"model.json"), "w") as json_file:
            #json_file.write(model_json)

        logger.debug(" -> save_model")

    def get_curr_folder(self):
        self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        curr_folder = create_path(self.dest_path, self.train_time)
        return curr_folder

    def set_model_checkpint(self):
        logger.debug(" <- set_model_checkpoint")
        curr_folder = self.get_curr_folder()
        out_model_path = os.path.join(curr_folder, self.weights_file_name)
        model_checkpoint = [ModelCheckpoint(out_model_path, monitor='val_loss',
                                            verbose=1, save_best_only=True)]
        if self.callbacks is None:
            self.callbacks = model_checkpoint
        else:
            self.callbacks = model_checkpoint + self.callbacks
        logger.debug(" -> set_model_checkpoint")
        return curr_folder


    def train_model(self,params_dict=None,saveflag=False):
        logger.debug(f" <- train_model, save_flag{saveflag}")
        if saveflag:
            self.save_model(params_dict,saveflag)
        self.fit_unet()

        logger.debug(" -> train_model")

    @staticmethod
    def load_model(src_path):
        #loaded_model=None
        params_dict = {}
        pretrained_weights = {}
        files = os.scandir(src_path)
        for file_entry in files:
            file_name_segments = file_entry.name.rsplit('.', 1)
            file_name = file_name_segments[0]
            file_extention = file_name_segments[-1]
            if file_entry.name == 'model_params.json':
                params_dict = load_json(file_entry.path)
            #elif file_entry.name == 'model.json':
                #loaded_model = model_from_json(file_entry.path)
                # load weights into new model


            elif file_extention == 'hdf5':
                pretrained_weights = file_entry.path

        #loaded_model.load_weights("model.h5")
        params_dict['pretrained_weights'] = pretrained_weights
        params_dict['train_time'] = os.path.basename(src_path)

        return params_dict
