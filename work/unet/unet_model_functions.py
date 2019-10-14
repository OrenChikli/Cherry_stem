import glob
# from tqdm import tqdm # this causes problems with kers progress bar in jupyter!!!
import logging
import re
from datetime import datetime

import numpy as np
from auxiliary.data_functions import *
from keras.models import load_model
from keras.optimizers import get as get_optimizer
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K

from work.unet.unet_callbacks import CustomTensorboardCallback, CustomModelCheckpoint
from work.unet.unet_model import unet

SESS_REGEX = r"(train_sess_)(\d+)(\.)"

STEPS_REGEX = r"(steps_)(\d+)(\.)"

HDF5_STEPS_REGEX = r"(steps_)(\d+)(\-)"

PARAMS_FILENAME = "model_params"
PARAMS_UPDATE_FORMAT = '.train_sess_{sess:02d}.steps_{steps:02d}.'

logger = logging.getLogger(__name__)

MODES_DICT = {'grayscale': 1, 'rgb': 3}  # translate for image dimensions
COLOR_TO_OPENCV = {'grayscale': 0, 'rgb': 1}


class ClarifruitUnet:

    def __init__(self, train_path, save_path, weights_file_name,
                 x_folder_name='image', y_folder_name='label',
                 data_gen_args=None, callbacks=None,
                 optimizer=None, optimizer_params=None, loss=None, metrics=None, model_checkpoint=None,
                 target_size=(256, 256), color_mode='rgb',
                 batch_size=10, epochs=5, steps_per_epoch=10,
                 validation_split=0.2, validation_steps=10,
                 train_time=None,
                 seed=1,
                 tensorboard_update_freq=100,
                 weights_update_freq=1000,
                 ontop_display_threshold=0.5):

        logger.debug(" <- __init__")

        K.clear_session()
        self.train_path = train_path
        self.x_folder_name = x_folder_name
        self.y_folder_name = y_folder_name
        self.weights_file_name = weights_file_name

        self.data_gen_args = data_gen_args
        self.target_size = tuple(target_size)
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

        if model_checkpoint is not None:
            self.model = load_model(model_checkpoint)
        else:
            self.model = None
            self.get_unet_model()

        logger.debug(" -> __init__")

    @staticmethod
    def train_val_generators(batch_size, src_path, folder, save_prefix,
                             input_aug_dict=None,
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
        :param input_aug_dict: a dictionary with the data augmentation parameters of the images
        :param save_prefix: if output images are saved, this is the prefix in the file names
        :param color_mode: how to load the images, options are "grayscale", "rgb", default is "grayscale"
        :param save_to_dir: path to save output images, if None nothing is saved, default is None
        :param target_size: pixel size of output images,default is (256,256)
        :param validation_split: size of the validation data set, default is 0.2
        :param seed: random seed used in image generation, default is 1
        :return: a flow_from_dictionary keras datagenerator
        """
        logger.debug(f" <- train_val_generators src:\n{src_path}")

        aug_dict = {'rescale': 1. / 255}  # always rescale the images to the model
        if input_aug_dict is not None:
            aug_dict.update(input_aug_dict)

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

    def clarifruit_train_val_generators(self, aug_flag=True):
        """
        a method to create train and validation data generators for the current instance
        :return:
        """
        logger.debug(f" <- clarifruit_train_val_generators")
        aug_dict = self.data_gen_args if aug_flag else None

        image_train_generator, image_val_generator = self.train_val_generators(batch_size=self.batch_size,
                                                                               src_path=self.train_path,
                                                                               folder=self.x_folder_name,
                                                                               input_aug_dict=aug_dict,
                                                                               color_mode=self.color_mode,
                                                                               save_prefix=self.x_folder_name,
                                                                               save_to_dir=self.save_to_dir,
                                                                               target_size=self.target_size,
                                                                               seed=self.seed,
                                                                               validation_split=self.validation_split)

        mask_train_generator, mask_val_generator = self.train_val_generators(batch_size=self.batch_size,
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
        logger.debug(f" -> clarifruit_train_val_generators")
        return train_generator, val_generator

    def test_generator(self, test_path):
        """
        create a generator which yield appropriate images be be used with the model's predict
        method, i.e reshapes the images and loads them in the appropriate color mode
        :param test_path:
        :return: img- an image in an apropriate dimentions for the unet model predict method
                 img_entry- the result of the os.scandir method, and object with the source image name and path
                 orig_shape- the original shape of the source image, to be used for reshaping the prediction back to
                             the source image size
        """
        logger.debug(" <-test_generator")

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

    def prediction_generator(self, test_path):
        """
        a method to yield predictions from the test path
        :param test_path: a path containing the test images
        :return: img_entry- the result of the os.scandir method, and object with the source image name and path
                 pred_raw_resised- a mask image, the prediction for the image
        """
        logger.info(f" generating prediction on files from {test_path}")

        logger.debug(" <- prediction_generator")
        test_gen = self.test_generator(test_path)
        for img, img_entry, orig_shape in test_gen:
            pred_raw = self.model.predict(img, batch_size=1)[0]
            pred_raw_resized = cv2.resize(pred_raw, orig_shape)
            yield img_entry, pred_raw_resized

    def prediction(self, test_path, dest_path):
        """
        a method to get predictions from a trained model of images in the test_path variable, and save the results to the
        path specified in the dest_path variable
        :param dest_path: the destination path to save he prediction results
        :param test_path: the path where the test data resides
        :return:
        """
        logger.info(f"prediction on files from {test_path}")

        logger.debug(" <- prediction")
        if self.train_time is None:
            self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        save_path = create_path(dest_path, self.train_time)
        save_path = create_path(save_path, 'raw_pred')
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

        logger.debug(" -> prediction")
        return save_path

    def get_unet_model(self):
        """
        load a unet model for the current instance
        :return:
        """
        logger.debug(" <- get_unet_model")

        # create optimizer instance
        config = {
            'class_name': self.optimizer,
            'config': self.optimizer_params}
        optimizer = get_optimizer(config)

        self.model = unet(optimizer=optimizer,
                          loss=self.loss,
                          metrics=self.metrics,
                          input_size=self.input_size)
        logger.debug(" -> get_unet_model")

    def fit_unet(self):
        """ fit a unet model for the current instance"""
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

    def save_model_params(self):
        logger.debug(" <- save_model_params")

        if self.train_time is None:
            self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        curr_folder = create_path(self.save_path, self.train_time)

        params_dict = self.get_model_params()
        if self.params_filepath is not None:
            file_params = load_json(self.params_filepath)
            if file_params != params_dict:  # cheking if the parametes for this session are diffrent
                self.session_number += 1

        curr_file_name = (PARAMS_FILENAME + PARAMS_UPDATE_FORMAT + 'json').format(sess=self.session_number,
                                                                                  steps=self.samples_seen)
        save_json(params_dict, curr_file_name, curr_folder)
        self.params_filepath = os.path.join(curr_folder, curr_file_name)

        logger.debug(" -> save_model_params")

    def get_model_params(self):
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
                          'session_number']

        for key in exclude_params:
            params_dict.pop(key)
        return params_dict

    def set_model_checkpint(self):
        """
        set the model checkpoint keras callbacks method for the current training session,
        where the model weights will be saved in folder assigned for the current session
        :param dest_path: the destination folder where the specific session will be saved to
        :return: the save folder for the current training session
        """
        logger.debug(" <- set_model_checkpoint")
        monitor = 'loss'

        if self.train_time is None:
            self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        curr_folder = create_path(self.save_path, self.train_time)

        keras_logs_path = create_path(curr_folder, self.keras_logs_folder)

        file_name = self.weights_file_name

        steps_file_name = file_name + PARAMS_UPDATE_FORMAT + '%s_{loss:.2f}.hdf5' % monitor
        steps_out_model_path = os.path.join(curr_folder, steps_file_name)

        steps_model_checkpoint = CustomModelCheckpoint(steps_out_model_path,
                                                       monitor='loss',
                                                       verbose=0,
                                                       save_best_only=False,
                                                       update_freq=self.weights_update_freq,
                                                       batch_size=self.batch_size,
                                                       save_weights_only=False,
                                                       samples_seen=self.samples_seen,
                                                       model_params_path=self.params_filepath,
                                                       session_n=self.session_number)

        # get some non augmented images for tensorboard visualizations
        _, val_gen_no_aug = self.clarifruit_train_val_generators(aug_flag=False)
        # TODO modify hardcoded values
        v_data = [next(val_gen_no_aug) for i in range(1000) if i % 200 == 0.0]

        image_history = CustomTensorboardCallback(log_dir=keras_logs_path,
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
        logger.debug(" -> set_model_checkpoint")
        return keras_logs_path

    def set_model_for_train(self):
        """
        train the unet model for current instance and save the results if possible
        :param params_dict: the parameters used to define the current instance
        :param dest_path: optional destination path to save the model
        :return:
        """
        logger.debug(" <- set_model_for_train")
        self.save_model_params()

        self.train_generator, self.val_generator = self.clarifruit_train_val_generators()
        keras_logs_path = self.set_model_checkpint()
        logger.debug(" -> set_model_for_train")

        return keras_logs_path

    @staticmethod
    def get_pattern(file, regex, group_n):
        ret = None
        pattern = re.search(regex, file)
        if pattern is not None:
            ret = int(pattern.group(group_n))
        return ret

    @staticmethod
    def get_file(src_path, steps, file_extention, regex):
        res = None
        func = ClarifruitUnet.get_pattern
        files_iterator = glob.iglob(os.path.join(src_path, f'*.{file_extention}'))
        sorted_file_names = sorted([(func(file, regex, 2), file) for file in files_iterator])
        for samples_seen, file in sorted_file_names:
            if samples_seen >= steps:
                res = file
                steps = samples_seen
                break

        return res, steps

    @staticmethod
    def load_model(src_path, steps=None):
        """
        load a pretrained model located in the src_path
        :param src_path: the path containing the pretrained model
        :return: the parameters of the model to be used later on
        """
        if steps is not None:
            json_file, _ = ClarifruitUnet.get_file(src_path, steps, 'json', STEPS_REGEX)
            hdf5_file, samples_seen = ClarifruitUnet.get_file(src_path, steps, 'hdf5', STEPS_REGEX)


        else:
            json_file = max(glob.iglob(os.path.join(src_path, '*.json')), key=os.path.getctime)
            hdf5_file = max(glob.iglob(os.path.join(src_path, '*.hdf5')), key=os.path.getctime)

            samples_seen = ClarifruitUnet.get_pattern(hdf5_file, STEPS_REGEX, 2)

        session_number = ClarifruitUnet.get_pattern(json_file, SESS_REGEX, 2)

        params_dict = load_json(json_file)
        params_dict['pretrained_weights'] = hdf5_file
        params_dict['train_time'] = os.path.basename(src_path)

        model = ClarifruitUnet(**params_dict)
        model.samples_seen = samples_seen
        model.params_filepath = json_file
        model.session_number = session_number

        return model

    def update_model(self, **kwargs):
        self.__dict__.update(kwargs)
