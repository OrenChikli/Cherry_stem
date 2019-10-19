from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
import cv2
from datetime import datetime
import glob
from work.auxiliary import data_functions
from keras.optimizers import get as get_optimizer
from keras.models import load_model

from work.auxiliary.decorators import Logger_decorator

from work.unet.unet_model import unet
from work.unet.unet_callbacks import CustomTensorboardCallback, \
    CustomModelCheckpoint

import logging
from tensorflow.python.keras import backend as K
import re

SESS_REGEX = r"(train_sess_)(\d+)(\.)"

STEPS_REGEX = r"(steps_)(\d+)(\.)"

HDF5_STEPS_REGEX = r"(steps_)(\d+)(\-)"

PARAMS_UPDATE_FORMAT = '.train_sess_{sess:02d}.steps_{steps:02d}.'

MODES_DICT = {'grayscale': 1, 'rgb': 3}  # translate for image dimensions
COLOR_TO_OPENCV = {'grayscale': 0, 'rgb': 1}

logger = logging.getLogger(__name__)
logger_decorator = Logger_decorator(logger)


class ClarifruitUnet:
    """
    a set of functions to use the unet model for image segmentation
    """

    @logger_decorator.debug_dec
    def __init__(self, train_path, save_path,
                 save_name='model',
                 pretrained_weights=None, checkpoint=None,

                 x_folder_name='image',
                 y_folder_name='label',

                 seed=1, data_gen_args=None, validation_split=0.2,
                 validation_steps=10,

                 callbacks=None,
                 optimizer=None, optimizer_params=None, loss=None, metrics=None,

                 target_size=(256, 256), color_mode='rgb',

                 batch_size=10, epochs=5, steps_per_epoch=10,

                 train_time=None,

                 tensorboard_update_freq=100, weights_update_freq=1000,
                 ontop_display_threshold=0.5,
                 save_weights_only=True):

        K.clear_session()
        self.save_name = save_name
        self.train_path = train_path
        self.x_folder_name = x_folder_name
        self.y_folder_name = y_folder_name

        self.save_name = save_name
        self.params_file_name = save_name + '_params'
        self.weights_file_name = save_name + '_weights'
        self.checkpoint_filename = save_name + '_ckpt'

        self.data_gen_args = data_gen_args
        self.target_size = target_size
        self.color_mode = color_mode
        self.input_size = (*target_size, MODES_DICT[color_mode])

        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.loss = loss
        self.metrics = metrics

        self.train_generator = None
        self.val_generator = None
        self.validation_split = validation_split
        self.seed = seed

        self.callbacks = callbacks
        self.train_time = train_time

        self.save_to_dir = None

        self.tensorboard_update_freq = tensorboard_update_freq
        self.weights_update_freq = weights_update_freq
        self.ontop_display_threshold = ontop_display_threshold
        self.save_path = save_path
        self.keras_logs_folder = 'keras_logs'

        self.samples_seen = 0
        self.session_number = 1
        self.params_filepath = None
        self.curr_folder = None

        self.save_weights_only = save_weights_only

        self.pretrained_weights = pretrained_weights
        self.checkpoint = checkpoint

        if checkpoint is not None:
            logger.info(f"loaing model from checkpoint: {checkpoint}")
            self.model = load_model(checkpoint)

        else:
            if pretrained_weights is not None:
                logger.info(f"loading model weights from {pretrained_weights}")
            self.model = None
            self.get_unet_model()

    @staticmethod
    @logger_decorator.debug_dec
    def train_val_generators(batch_size, src_path, folder, save_prefix,
                             input_aug_dict=None,
                             color_mode="grayscale",
                             save_to_dir=None,
                             target_size=(256, 256),
                             validation_split=0.2,
                             seed=1):
        """

        Create a datagen generator with train validation split.
        :param batch_size:int, the batch size of each step.
        :param src_path:str, the path to the data.
        :param folder:str, the name of the folder in the data_path.
        :param input_aug_dict:optional, dict, a dictionary with the data
         augmentation parameters of the images.
        :param save_prefix:str, if output images are saved, this is the prefix
         in the file names.
        :param color_mode:optional,str, how to load the images, options are
         "grayscale", "rgb", default is "grayscale".
        :param save_to_dir:optional,str, path to save output images,
         if None- nothing is saved, default is None.
        :param target_size:optional,tuple, pixel size of output images,
        default is (256,256).
        :param validation_split:optional,float, size of the validation data set,
         default is 0.2.
        :param seed: optional,int, random seed used in image generation,
         default is 1.
        :return: a flow_from_dictionary keras datagenerator
        """

        aug_dict = {'rescale': 1. / 255}  # always rescale the images to
        # the model
        if input_aug_dict is not None:
            aug_dict.update(input_aug_dict)

        datagen = ImageDataGenerator(**aug_dict,
                                     validation_split=validation_split)

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

    @logger_decorator.debug_dec
    def clarifruit_train_val_generators(self, aug_flag=True):
        """
        a method to create train and validation data generators for the current
        instance
        :param aug_flag: bool, optional, whether to use the augdict parameters,
        used for tensorboard visualizations of several validation set images
        :return:
        """

        aug_dict = self.data_gen_args if aug_flag else None

        image_train_generator, image_val_generator = self.train_val_generators(
            batch_size=self.batch_size,
            src_path=self.train_path,
            folder=self.x_folder_name,
            input_aug_dict=aug_dict,
            color_mode=self.color_mode,
            save_prefix=self.x_folder_name,
            save_to_dir=self.save_to_dir,
            target_size=self.target_size,
            seed=self.seed,
            validation_split=self.validation_split)

        mask_train_generator, mask_val_generator = self.train_val_generators(
            batch_size=self.batch_size,
            src_path=self.train_path,
            folder=self.y_folder_name,
            input_aug_dict=aug_dict,
            color_mode='grayscale',
            save_prefix=self.y_folder_name,
            save_to_dir=self.save_to_dir,
            target_size=self.target_size,
            seed=self.seed,
            validation_split=self.validation_split)

        train_generator = zip(image_train_generator, mask_train_generator)
        val_generator = zip(image_val_generator, mask_val_generator)

        return train_generator, val_generator

    @logger_decorator.debug_dec
    def test_generator(self, test_path):
        """
        create a generator which yield appropriate images be be used with the
        model's predict method, i.e reshapes the images and loads them in the
         appropriate color mode
        :param test_path:
        :return: img- an image in an apropriate dimentions for the unet model
         predict method
                 img_entry- the result of the os.scandir method, and object with
                  the source image name and path
                 orig_shape- the original shape of the source image, to be used
                  for reshaping the prediction back to
                             the source image size
        """

        img_list = os.scandir(test_path)
        for img_entry in img_list:

            img = cv2.imread(img_entry.path, COLOR_TO_OPENCV[self.color_mode])
            if img.shape[-1] == 3:
                orig_shape = img.shape[-2::-1]
            else:
                orig_shape = img.shape[::-1]

            img = img / 255
            img = cv2.resize(img, self.target_size)
            if self.color_mode == "grayscale":
                img = np.reshape(img, img.shape + (1,))
            img = np.reshape(img, (1,) + img.shape)
            yield img, img_entry, orig_shape

    @logger_decorator.debug_dec
    def prediction_generator(self, test_path):
        """
        a method to yield predictions from the test path
        :param test_path: a path containing the test images
        :return: img_entry- the result of the os.scandir method, and object with
                            the source image name and path
                 pred_raw_resized- a mask image, the prediction for the image
        """
        logger.info(f" generating prediction on files from {test_path}")

        test_gen = self.test_generator(test_path)
        for img, img_entry, orig_shape in test_gen:
            pred_raw = self.model.predict(img, batch_size=1)[0]
            pred_raw_resized = cv2.resize(pred_raw, orig_shape)
            yield img_entry, pred_raw_resized

    @logger_decorator.debug_dec
    def prediction(self, test_path, dest_path):
        """
        a method to get predictions from a trained model of images in the
        test_path variable, and save the results to the path specified in the
        dest_path variable
        :param dest_path: the destination path to save he prediction results
        :param test_path: the path where the test data resides
        :return:
        """
        logger.info(f"prediction on files from {test_path}")

        if self.train_time is None:
            self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        save_path = data_functions.create_path(dest_path, self.train_time)
        save_path = data_functions.create_path(save_path, 'raw_pred')
        logger.info(f"saving predictions to {save_path}")
        # saving the src_path of the current files
        with open(os.path.join(save_path, "src_path.txt"), 'w') as f:
            f.write(test_path)

        test_gen = self.test_generator(test_path)
        for img, img_entry, orig_shape in test_gen:
            pred_raw = self.model.predict(img, batch_size=1)[0]
            pred_raw_resized = cv2.resize(pred_raw, orig_shape)

            file_name = img_entry.name.rsplit('.', 1)[0] + '.npy'
            npy_file_save_path = os.path.join(save_path, file_name)
            np.save(npy_file_save_path, pred_raw_resized, allow_pickle=True)

            pred_image = (255 * pred_raw_resized).astype(np.uint8)
            cv2.imwrite(os.path.join(save_path, img_entry.name), pred_image)

        return save_path

    @logger_decorator.debug_dec
    def get_unet_model(self):
        """
        load a unet model for the current instance
        :return:
        """
        # create optimizer instance
        config = {
            'class_name': self.optimizer,
            'config': self.optimizer_params}
        optimizer = get_optimizer(config)

        self.model = unet(optimizer=optimizer,
                          loss=self.loss,
                          metrics=self.metrics,
                          input_size=self.input_size,
                          pretrained_weights=self.pretrained_weights)

    @logger_decorator.debug_dec
    def fit_unet(self):
        """
        fit a unet model for the current instance
        :return:
        """
        logger.info(f"Training model with optimizer:{self.optimizer}")
        logger.info(f"Optimizer params: {self.optimizer_params}")
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.val_generator,
            validation_steps=self.validation_steps,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=1)

    @logger_decorator.debug_dec
    def save_model_params(self):
        """
        save the current ClarifruitUnet instance parameters to a json file
        :return:
        """
        params_dict = self.get_model_params()
        if self.params_filepath is not None:
            file_params = data_functions.load_json(self.params_filepath)
            if file_params != params_dict:  # cheking if the parametes for this
                # session are diffrent then those
                # in the source file
                self.session_number += 1

        curr_file_name = (
                self.params_file_name + PARAMS_UPDATE_FORMAT + 'json').format(
            sess=self.session_number,
            steps=self.samples_seen)

        data_functions.save_json(params_dict, curr_file_name, self.curr_folder)
        self.params_filepath = os.path.join(self.curr_folder, curr_file_name)

    @logger_decorator.debug_dec
    def get_model_params(self):
        """
        get a dictionary of the model parameters to be saved, removes
        unnecessary parameters
        :return: a dictionary of parameters
        """
        params_dict = vars(self).copy()
        exclude_params = ['input_size',
                          'model',
                          'train_generator',
                          'val_generator',
                          'callbacks',
                          'save_to_dir',
                          'keras_logs_folder',
                          'samples_seen',
                          'params_filepath',
                          'session_number',
                          'params_file_name',
                          'weights_file_name',
                          'checkpoint_filename',
                          'curr_folder'
                          ]

        for key in exclude_params:
            params_dict.pop(key)
        return params_dict

    @logger_decorator.debug_dec
    def set_model_checkpint(self):
        """
        set the model checkpoint keras callbacks method for the current training
        session, where the model weights will be saved in folder assigned for
        the current session
        :return: the save folder for the current training session
        """

        keras_logs_path = data_functions.create_path(self.curr_folder,
                                                     self.keras_logs_folder)

        file_name = self.weights_file_name if self.save_weights_only \
            else self.checkpoint_filename

        steps_file_name = file_name + \
                          PARAMS_UPDATE_FORMAT + \
                          'loss_{loss:.4f}.hdf5'

        steps_out_model_path = os.path.join(self.curr_folder, steps_file_name)

        steps_model_checkpoint = CustomModelCheckpoint(
            steps_out_model_path,
            monitor='loss',
            verbose=1,
            save_best_only=False,
            update_freq=self.weights_update_freq,
            batch_size=self.batch_size,
            save_weights_only=self.save_weights_only,
            samples_seen=self.samples_seen,
            model_params_path=self.params_filepath,
            session_n=self.session_number)

        # get some non augmented images for tensorboard visualizations
        _, val_gen_no_aug = self.clarifruit_train_val_generators(aug_flag=False)

        # TODO modify hardcoded values
        v_data = [next(val_gen_no_aug) for i in range(1000) if i % 200 == 0.0]

        image_history = CustomTensorboardCallback(
            log_dir=keras_logs_path,
            batch_size=self.batch_size,
            histogram_freq=0,
            write_graph=True,
            update_freq=self.tensorboard_update_freq,
            data=v_data,
            threshold=self.ontop_display_threshold,
            samples_seen=self.samples_seen)

        callbacks = [image_history, steps_model_checkpoint]

        if self.callbacks is None:
            self.callbacks = callbacks
        else:
            self.callbacks = callbacks + self.callbacks

        return keras_logs_path

    @logger_decorator.debug_dec
    def set_model_for_train(self):
        """
        prepare the instance for training.
        set the destenation folder for saving the results, save the model
        parameters set the model checkpoints and get the model generators
        :return: the path of the keras logs to be used with tensorboard
        """
        if self.train_time is None:
            self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.curr_folder = data_functions.create_path(
            self.save_path, self.train_time)
        logger.info(f"training results will be stored in: {self.curr_folder}")

        self.save_model_params()
        self.train_generator, self.val_generator = \
            self.clarifruit_train_val_generators()
        keras_logs_path = self.set_model_checkpint()

        return keras_logs_path

    @staticmethod
    @logger_decorator.debug_dec
    def get_pattern(src_string, regex):
        """
        use a regex expression with 3 capturing groups to return the 2 group
        if the pattern exsists in the src_string
        :param src_string: str, a string to be searched for the pattern
        :param regex: a regular expression which has 3 capturing groups
        :return: the second capturing group if the pattern is found
        """
        ret = None
        pattern = re.search(regex, src_string)
        if pattern is not None:
            ret = int(pattern.group(2))
        return ret

    @classmethod
    @logger_decorator.debug_dec
    def get_file_via_steps(cls, src_path, steps, file_extention, regex):
        """
        load a file which has a number of steps in it's file name, which are
        greater or equal to the given steps
        :param src_path: the source path to the files
        :param steps: the number of steps to be searched
        :param file_extention: the extention type of the files to be searced
        :param regex: the regular expression with 3 capuring groups where the
        2 group corresponds to the number of steps
        :return:
        """
        res = None
        func = cls.get_pattern
        files_iterator = glob.iglob(
            os.path.join(src_path, f'*.{file_extention}'))
        sorted_file_names = sorted(
            [(func(file, regex, 2), file) for file in files_iterator])
        for samples_seen, file in sorted_file_names:
            if samples_seen >= steps:
                res = file
                steps = samples_seen
                break
        logger.warning("couldnt find files for the specified number of steps,"
                       "loading the latest files instead")

        return res, steps

    @classmethod
    @logger_decorator.debug_dec
    def load_model(cls, src_path, update_dict=None, steps=None):
        """
        load a pretrained model located in the src_path
        :param src_path: the path containing the pretrained model
        :param update_dict: optional, dict, a dictionary of parameter used to
        modifiey the loaded model
        :param steps:optional,int, the steps from which to load the model,
        if None, the latest weights and parameters are loaded, default None

        :return:  a ClarifruitUnet instance
        """

        if steps is not None:
            json_file, _ = cls.get_file_via_steps(src_path, steps, 'json', STEPS_REGEX)
            hdf5_file, samples_seen = cls.get_file_via_steps(src_path, steps, 'hdf5',
                                                             STEPS_REGEX)


        else:
            json_file = max(glob.iglob(os.path.join(src_path, '*.json')),
                            key=os.path.getctime)
            hdf5_file = max(glob.iglob(os.path.join(src_path, '*.hdf5')),
                            key=os.path.getctime)

            samples_seen = cls.get_pattern(hdf5_file, STEPS_REGEX, 2)
            samples_seen = samples_seen if samples_seen is not None else 0

        session_number = cls.get_pattern(hdf5_file, SESS_REGEX, 2)
        session_number = session_number if session_number is not None else 1

        params_dict = data_functions.load_json(json_file)

        params_dict['pretrained_weights'] = hdf5_file

        #TODO: try to rearange loading weights
        # if 'weights' in os.path.basename(hdf5_file):
        #     params_dict['pretrained_weights'] = hdf5_file
        # else:
        #     params_dict['checkpoint'] = hdf5_file

        params_dict['train_time'] = os.path.basename(src_path)
        if update_dict is not None:
            if 'pretrained_weights' or 'checkpoint' in update_dict:
                params_dict['pretrained_weights'] = None
                params_dict['checkpoint'] = None
            params_dict.update(update_dict)

        model = ClarifruitUnet(**params_dict)
        logger.info(f"continuing training from {os.path.basename(hdf5_file)}")

        setattr(model, 'samples_seen', samples_seen)
        setattr(model, 'params_filepath', json_file)
        setattr(model, 'session_number', session_number)

        return model

    @logger_decorator.debug_dec
    def update_model(self, **kwargs):
        """
        update the current instance to have new parameter given in kwargs
        :param kwargs: the new update parameters
        :return:
        """
        self.__dict__.update(kwargs)
        opt_params = ['optimizer_params', 'optimizer']
        if any(item in kwargs.keys() for item in opt_params):
            self.get_unet_model()
