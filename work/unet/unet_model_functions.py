from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from datetime import datetime
from keras.callbacks import ModelCheckpoint, TensorBoard
from auxiliary.data_functions import *
from keras.optimizers import *
from work.unet.unet_model import unet
from work.unet.unet_callbacks import ImageHistory
# from tqdm import tqdm # this causes problems with kers progress bar in jupyter!!!
import logging

logger = logging.getLogger(__name__)

MODES_DICT = {'grayscale': 1, 'rgb': 3}  # translate for image dimensions
COLOR_TO_OPENCV = {'grayscale': 0, 'rgb': 1}
OPTIMIZER_DICT = {'Adam': Adam, 'adagrad': adagrad}


class ClarifruitUnet:

    def __init__(self, train_path, weights_file_name,
                 x_folder_name='image', y_folder_name='label',
                 data_gen_args=None, callbacks=None,
                 optimizer=None, optimizer_params=None, loss=None, metrics=None, pretrained_weights=None,
                 target_size=(256, 256), color_mode='rgb',
                 batch_size=10, epochs=5, steps_per_epoch=10,
                 valdiation_split=0.2, validation_steps=10,
                 train_time=None,
                 steps=0):

        logger.debug(" <- __init__")

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
        self.steps=steps

        self.get_unet_model()

        logger.debug(" -> __init__")

    @staticmethod
    def custom_generator(batch_size, src_path, folder, aug_dict, save_prefix,
                         color_mode="grayscale",
                         save_to_dir=None,
                         target_size=(256, 256),
                         seed=1,
                         validation_split=0.0,
                         subset=None
                         ):
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

        datagen = ImageDataGenerator(**aug_dict, validation_split=validation_split)

        gen = datagen.flow_from_directory(
            src_path,
            classes=[folder],
            class_mode=None,
            color_mode=color_mode,
            target_size=target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            seed=seed,
            subset=subset)

        logger.debug(f"-> custom_generator, src:\n{src_path}")
        return gen

    # @staticmethod
    # def train_val_generators(batch_size, src_path, folder, aug_dict, save_prefix,
    #                          color_mode="grayscale",
    #                          save_to_dir=None,
    #                          target_size=(256, 256),
    #                          validation_split=0.2,
    #                          seed=1):
    #     """
    #
    #     Create a datagen generator with  train validation split
    #     :param batch_size: the batch size of each step
    #     :param src_path: the path to the data
    #     :param folder: the name of the folder in the data_path
    #     :param aug_dict: a dictionary with the data augmentation parameters of the images
    #     :param save_prefix: if output images are saved, this is the prefix in the file names
    #     :param color_mode: how to load the images, options are "grayscale", "rgb", default is "grayscale"
    #     :param save_to_dir: path to save output images, if None nothing is saved, default is None
    #     :param target_size: pixel size of output images,default is (256,256)
    #     :param validation_split: size of the validation data set, default is 0.2
    #     :param seed: random seed used in image generation, default is 1
    #     :return: a flow_from_dictionary keras datagenerator
    #     """
    #     logger.debug(f" <- train_val_generators src:\n{src_path}")
    #     datagen = ImageDataGenerator(**aug_dict, validation_split=validation_split)
    #
    #     train_gen = ClarifruitUnet.custom_generator(src_path=src_path,
    #                                                 aug_dict=aug_dict,
    #                                                 folder=folder,
    #                                                 color_mode=color_mode,
    #                                                 target_size=target_size,
    #                                                 batch_size=batch_size,
    #                                                 save_to_dir=save_to_dir,
    #                                                 save_prefix=save_prefix,
    #                                                 seed=seed,
    #                                                 subset='training')
    #
    #     val_gen = ClarifruitUnet.custom_generator(src_path=src_path,
    #                                                 aug_dict=aug_dict,
    #                                                 folder=folder,
    #                                                 color_mode=color_mode,
    #                                                 target_size=target_size,
    #                                                 batch_size=batch_size,
    #                                                 save_to_dir=save_to_dir,
    #                                                 save_prefix=save_prefix,
    #                                                 seed=seed,
    #                                                 subset='validation')
    #
    #     logger.debug(f" -> train_val_generators src:\n{src_path}")
    #     return train_gen, val_gen

    @staticmethod
    def train_val_generators(batch_size, src_path, folder,  save_prefix,
                             aug_dict = None,
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
        if aug_dict is not None:
            datagen = ImageDataGenerator(**aug_dict, validation_split=validation_split)

        else:
            datagen = ImageDataGenerator(validation_split=validation_split)

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

    def clarifruit_train_val_generators(self,aug_flag=True):
        """
        a method to create train and validation data generators for the current instance
        :return:
        """
        logger.debug(f" <- clarifruit_train_val_generators")
        aug_dict = self.data_gen_args if aug_flag else None
        image_train_generator, image_val_generator = self.train_val_generators(batch_size=self.batch_size,
                                                                               src_path=self.train_path,
                                                                               folder=self.x_folder_name,
                                                                               aug_dict=aug_dict,
                                                                               color_mode=self.color_mode,
                                                                               save_prefix=self.x_folder_name,
                                                                               save_to_dir=self.save_to_dir,
                                                                               target_size=self.target_size,
                                                                               seed=self.seed,
                                                                               validation_split=self.validation_split)

        mask_train_generator, mask_val_generator = self.train_val_generators(batch_size=self.batch_size,
                                                                             src_path=self.train_path,
                                                                             folder=self.y_folder_name,
                                                                             aug_dict=aug_dict,
                                                                             color_mode='grayscale',
                                                                             save_prefix=self.y_folder_name,
                                                                             save_to_dir=self.save_to_dir,
                                                                             target_size=self.target_size,
                                                                             seed=self.seed,
                                                                             validation_split=self.validation_split)

        train_generator = zip(image_train_generator, mask_train_generator)
        val_generator = zip(image_val_generator, mask_val_generator)
        logger.debug(f" -> clarifruit_train_val_generators")
        return train_generator,val_generator


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
        self.model = unet(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics,
                          pretrained_weights=self.pretrained_weights,
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

    def save_model_params(self, dest_path, params_dict):
        logger.debug(" <- save_model")
        if self.train_time is None:
            self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        curr_folder = create_path(dest_path, self.train_time)

        save_dict = params_dict.copy()
        if 'callbacks' in save_dict:  # callbacks are not hashable, cant save to json
            save_dict.pop('callbacks')
        save_json(save_dict, "model_params.json", curr_folder)

        logger.debug(" -> save_model")


    def set_model_checkpint(self, dest_path):
        """
        set the model checkpoint keras callbacks method for the current training session,
        where the model weights will be saved in folder assigned for the current session
        :param dest_path: the destination folder where the specific session will be saved to
        :return: the save folder for the current training session
        """
        logger.debug(" <- set_model_checkpoint")

        if self.train_time is None:
            self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        curr_folder = create_path(dest_path, self.train_time)

        keras_logs_path = create_path(curr_folder, 'keras_logs')
        tensorboard_callback = TensorBoard(log_dir=keras_logs_path,
                                           histogram_freq=0,
                                           write_graph=True,
                                           write_images=True,
                                           update_freq=100)

        out_model_path = os.path.join(curr_folder, self.weights_file_name)
        model_checkpoint = ModelCheckpoint(out_model_path, monitor='loss',
                                           verbose=1, save_best_only=True)

        _,val_gen_no_aug = self.clarifruit_train_val_generators(aug_flag=False)

        v_data = next(val_gen_no_aug)
        #TODO: modify steps
        draw_interval = 100
        last_step = self.steps/ draw_interval
        image_history = ImageHistory(tensor_board_dir=keras_logs_path,
                                     data=v_data,
                                     last_step=last_step,
                                     draw_interval=draw_interval)

        callbacks = [tensorboard_callback, image_history, model_checkpoint]
        if self.callbacks is None:
            self.callbacks = callbacks
        else:
            self.callbacks = callbacks + self.callbacks
        logger.debug(" -> set_model_checkpoint")
        return curr_folder

    def train_model(self, params_dict, dest_path=None):
        """
        train the unet model for current instance and save the results if possible
        :param params_dict: the parameters used to define the current instance
        :param dest_path: optional destination path to save the model
        :return:
        """
        logger.debug(f" <- train_model")
        if dest_path is not None:
            self.save_model_params(dest_path=dest_path, params_dict=params_dict)

        self.train_generator,self.val_generator = self.clarifruit_train_val_generators()
        self.set_model_checkpint(dest_path=dest_path)
        self.fit_unet()

        logger.debug(" -> train_model")

    @staticmethod
    def load_model(src_path):
        """
        load a pretrained model located in the src_path
        :param src_path: the path containing the pretrained model
        :return: the parameters of the model to be used later on
        """

        params_dict = {}
        pretrained_weights = {}
        files = os.scandir(src_path)
        for file_entry in files:
            file_name_segments = file_entry.name.rsplit('.', 1)
            file_name = file_name_segments[0]
            file_extention = file_name_segments[-1]
            if file_entry.name == 'model_params.json':
                params_dict = load_json(file_entry.path)

            elif file_extention == 'hdf5':
                pretrained_weights = file_entry.path

        params_dict['pretrained_weights'] = pretrained_weights
        params_dict['train_time'] = os.path.basename(src_path)

        return params_dict
