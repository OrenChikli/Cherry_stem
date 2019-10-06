import os
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from auxiliary import data_functions
import shutil
from datetime import datetime
import logging
from sklearn.preprocessing import normalize


logger = logging.getLogger(__name__)

SKLEARN_CLASSIFIERS = {'LogisticRegression': LogisticRegression}


class StemHistClassifier:

    def __init__(self, train_path,hist_type='bgr',threshold='0.4'):
        logger.debug(" <- init")

        self.train_path = train_path
        self.hist_type=hist_type
        self.threshold=threshold

        self.train_list = self.load_train() if train_path is not None else None
        self.model = None
        self.train_time = None
        self.save_path=None
        logger.debug(" -> init")

    def load_train(self):
        logger.debug(" <- load_train")
        logger.debug(f"loading train data from:{self.train_path}")
        ret_list = []

        for label_folder in os.scandir(self.train_path):
            hist_folder = os.path.join(label_folder.path, f'{self.hist_type}_histograms')
            for hist_entry in os.scandir(hist_folder):
                ret_list.append((hist_entry, label_folder.name))

        random.shuffle(ret_list)
        logger.debug(" -> load_train")

        return ret_list
    @staticmethod
    def return_hist(hist,hist_type):
        if hist_type == 'bgr':
            hist = np.squeeze(hist, axis=2)
            hist = normalize(hist).flatten()
        else:
            hist = normalize(hist[0])
            hist = np.squeeze(hist, axis=1)
        return hist


    def data_iterator(self):
        for item_entry,item_label in self.train_list:
            hist = np.load(item_entry.path)
            hist = normalize(hist).flatten()
            yield hist, item_label


    def test_data_iterator(self,test_path):
        for item_entry in os.scandir(test_path):
            hist = np.load(item_entry.path)
            hist = normalize(hist).flatten().reshape(1,-1)
            yield item_entry.name, hist



    def train_model(self,save_path, model_name, **model_kwargs):
        logger.debug(" <- train_model")
        self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logger.info(f"model training time {self.train_time}")

        model = SKLEARN_CLASSIFIERS[model_name](**model_kwargs)

        logger.debug(" starting model_fit")
        x_train, y_train = zip(*self.data_iterator())
        x_train = np.array(x_train)
        model.fit(x_train,y_train)
        self.model = model

        self.save_model(model, model_kwargs, model_name, save_path)
        logger.debug(" -> train_model")

    def save_model(self, model, model_kwargs, model_name, save_path):

        save_path = data_functions.create_path(save_path, self.train_time)
        save_path = data_functions.create_path(save_path, 'saved_model')

        extractions_params = dict(threshold=self.threshold,hist_type=self.hist_type)
        data_functions.save_json(extractions_params,"extraction_params.json",save_path)

        data_functions.save_json(model_kwargs, f"{model_name}_input_params.json", save_path)
        data_functions.save_pickle(model, "trained_model.pickle", save_path)

    def model_predict(self,test_path,save_path, orig_images_path,img_extention='.jpg'):
        logger.debug(" <- model_predict")
        save_path = data_functions.create_path(save_path, self.train_time)
        for name, x in self.test_data_iterator(test_path):
            curr_name = name.rsplit('.',1)[0]+img_extention
            curr_img_path = os.path.join(orig_images_path, curr_name)
            pred = self.model.predict(x)[0]

            curr_save_path = data_functions.create_path(save_path, pred)
            _ = shutil.copy(curr_img_path, curr_save_path)
        logger.debug(" -> model_predict")

def get_pred_via_list(src_list,save_path, orig_images_path,img_extention='.jpg'):
    save_path = data_functions.create_path(save_path, "from_list")
    for name,pred in src_list:
        curr_name = name.rsplit('.',1)[0]+img_extention
        curr_img_path = os.path.join(orig_images_path, curr_name)
        curr_save_path = data_functions.create_path(save_path, str(pred))
        _ = shutil.copy(curr_img_path, curr_save_path)