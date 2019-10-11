import os
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from auxiliary import data_functions
import shutil
from datetime import datetime
import logging
from sklearn.preprocessing import normalize

from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pandas as pd
from sklearn.utils import shuffle

from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow.compat.v1.logging as tf_logging  # to stop tensorflow from displaying depracetion messages

tf_logging.set_verbosity(tf_logging.ERROR)

logger = logging.getLogger(__name__)

SKLEARN_CLASSIFIERS = {'LogisticRegression': LogisticRegression,
                       'RandomForestClassifier':RandomForestClassifier,
                       'XGBClassifier':XGBClassifier}


class StemHistClassifier:

    def __init__(self, train_path,label_col,drop_cols=None,hist_type='bgr',threshold='0.4'):
        logger.debug(" <- init")


        self.threshold=threshold
        self.hist_type = hist_type
        self.train_df = self.load_data(train_path,hist_type)
        self.label_col=label_col
        self.drop_cols=drop_cols

        self.model = None
        self.train_time = None
        self.save_path=None
        logger.debug(" -> init")

    @staticmethod
    def load_npy_data(src_path):
        df = None
        name_list = []
        for i, file_entry in enumerate(os.scandir(src_path)):
            if file_entry.name.endswith('.npy'):
                file = normalize(np.load(file_entry.path)).flatten()
                name = file_entry.name.rsplit('.', 1)[0]
                name_list.append(name)
                if df is None:
                    df = pd.DataFrame(file)
                else:
                    df[i] = file

        df = df.T
        df.columns = df.columns.astype(str)
        df.insert(0, "file_name", name_list)

        return df
    @staticmethod
    def load_data(path,hist_type):
        logger.debug(" <- load_data")
        logger.debug(f"loading train data from:{path}")
        ret_df = pd.DataFrame()

        for label_folder in os.scandir(path):
            hist_folder = os.path.join(label_folder.path, f'{hist_type}_histograms')
            curr_df = StemHistClassifier.load_npy_data(hist_folder)
            curr_df['label'] =label_folder.name
            ret_df = pd.concat((ret_df,curr_df))

        ret_df = shuffle(ret_df)

        logger.debug(" -> load_data")

        return ret_df

    @staticmethod
    def return_hist(hist,hist_type):
        if hist_type == 'bgr':
            hist = np.squeeze(hist, axis=2)
            hist = normalize(hist).flatten()
        else:
            hist = normalize(hist[0])
            hist = np.squeeze(hist, axis=1)
        return hist

    @staticmethod
    def from_list_data_generator(src_list):
        for item_entry,item_label in src_list:
            hist = np.load(item_entry.path)
            hist = normalize(hist).flatten()
            name = item_entry.name.rsplit('.',1)[0]
            yield hist, item_label,name



    @staticmethod
    def test_data_iterator(test_path):
        for item_entry in os.scandir(test_path):
            hist = np.load(item_entry.path)
            hist = normalize(hist).flatten().reshape(1,-1)
            yield item_entry.name, hist



    def train_model(self,save_path, model_name, **model_kwargs):
        logger.debug(" <- train_model")
        self.model = SKLEARN_CLASSIFIERS[model_name](**model_kwargs)
        x_train = self.train_df.drop([self.drop_cols,self.label_col],axis=1)
        y_train = self.train_df[self.label_col]
        logger.debug(" starting model_fit")
        self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logger.info(f"model training time {self.train_time}")
        self.model.fit(x_train,y_train)

        self.save_model(model_kwargs, model_name, save_path)
        logger.debug(" -> train_model")

    def save_model(self, model_kwargs, model_name, save_path):

        save_path = data_functions.create_path(save_path, self.train_time)
        save_path = data_functions.create_path(save_path, 'saved_model')

        extractions_params = dict(threshold=self.threshold,hist_type=self.hist_type)
        data_functions.save_json(extractions_params,"extraction_params.json",save_path)

        data_functions.save_json(model_kwargs, f"{model_name}_input_params.json", save_path)
        data_functions.save_pickle(self.model, "trained_model.pickle", save_path)



    def model_predict(self,test_path,save_path, orig_images_path,img_extention='.jpg'):
        logger.debug(" <- model_predict")
        save_path = data_functions.create_path(save_path, self.train_time)

        test_df = self.load_data(test_path,self.hist_type)
        x_test = test_df.drop([self.drop_cols, self.label_col], axis=1)
        y_test = test_df[self.label_col]
        y_pred = self.model.predict(x_test)
        for i in range(len(y_pred)):
            x_name = test_df.iloc[i]['file_name']
            img_name = x_name+img_extention
            curr_img_path = os.path.join(orig_images_path, img_name)
            curr_save_path = data_functions.create_path(save_path, y_pred[i])
            _ = shutil.copy(curr_img_path, curr_save_path)

        report = classification_report(y_test,y_pred)
        with open(os.path.join(save_path,"classification_report.txt"),'w') as f:
            f.write(report)
        print(report)
        logger.debug(" -> model_predict")

def get_pred_via_list(src_list,save_path, orig_images_path,img_extention='.jpg'):
    save_path = data_functions.create_path(save_path, "from_list")
    for name,pred in src_list:
        curr_name = name.rsplit('.',1)[0]+img_extention
        curr_img_path = os.path.join(orig_images_path, curr_name)
        curr_save_path = data_functions.create_path(save_path, str(pred))
        _ = shutil.copy(curr_img_path, curr_save_path)


def dl_model():



    def class_model(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=('accuracy'),
             pretrained_weights=None, input_size=(256, 256, 1)):
        """
        an impelemntation of the unet model, taken from https://github.com/zhixuhao/unet
        :param optimizer:keras optimizer to use in the model
        :param loss: keras loss function
        :param metrics: metrics list for
        :param pretrained_weights: path to possible pretrained weights which can be loaded into the model
        :param input_size: the dimensions of the input images, defualt is (256,256,1) images
        :return:
        """
        logger.debug(f"<- unet model with input_size={input_size} andpretraind_weights={pretrained_weights} ")
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        dense1 = Dense(4,activation='softmax',kernel_initializer='he_normal')(drop5)

        model = Model(inputs=inputs, outputs=dense1)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if pretrained_weights:
            model.load_weights(pretrained_weights)

        return model
