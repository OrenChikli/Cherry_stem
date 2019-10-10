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
import xgboost as xgb

logger = logging.getLogger(__name__)

SKLEARN_CLASSIFIERS = {'LogisticRegression': LogisticRegression,
                       'RandomForestClassifier':RandomForestClassifier,
                       'Xgboost':xgb}


class StemHistClassifier:

    def __init__(self, train_path,hist_type='bgr',threshold='0.4'):
        logger.debug(" <- init")

        self.hist_type=hist_type
        self.threshold=threshold

        self.train_list = self.load_data(train_path) if train_path is not None else None
        self.model = None
        self.train_time = None
        self.save_path=None
        logger.debug(" -> init")

    def load_data(self,path):
        logger.debug(" <- load_data")
        logger.debug(f"loading train data from:{path}")
        ret_list = []

        for label_folder in os.scandir(path):
            hist_folder = os.path.join(label_folder.path, f'{self.hist_type}_histograms')
            for hist_entry in os.scandir(hist_folder):
                ret_list.append((hist_entry, label_folder.name))

        random.shuffle(ret_list)
        logger.debug(" -> load_data")

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
        self.train_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logger.info(f"model training time {self.train_time}")



        logger.debug(" starting model_fit")
        if model_name =='Xgboost':
            self.model = SKLEARN_CLASSIFIERS[model_name]

            X_train,y_train,name= zip(*self.from_list_data_generator(self.train_list))
            train_data = xgb.DMatrix(X_train, label=y_train)
            param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
            num_round = 2
            self.model.train= xgb.train(param, train_data, num_round)
        else:
            self.model = SKLEARN_CLASSIFIERS[model_name](**model_kwargs)
            x_train, y_train,_ = zip(*self.from_list_data_generator(self.train_list))
            x_train = np.array(x_train)
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

    def use_xgb(self,test_path,save_path, orig_images_path,img_extention='.jpg'):
        data,_ = zip(*self.from_list_data_generator(self.train_list))
        dtrain = xgb.DMatrix(data)
        # specify parameters via map
        param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
        num_round = 2
        self.model = xgb.train(param, dtrain, num_round)
        self.save_model(model, model_kwargs, model_name, save_path)

    def model_predict(self,test_path,save_path, orig_images_path,img_extention='.jpg'):
        logger.debug(" <- model_predict")
        save_path = data_functions.create_path(save_path, self.train_time)

        test_list = self.load_data(test_path)
        y_list =[]
        pred_list= []
        for x_test, y_test,x_name in self.from_list_data_generator(test_list):
            y_list.append(y_test)
            x_test = np.array(x_test).reshape(1,-1)
            y_pred = self.model.predict(x_test)[0]
            pred_list.append(y_pred)
            img_name = x_name+img_extention
            curr_img_path = os.path.join(orig_images_path, img_name)
            curr_save_path = data_functions.create_path(save_path, y_pred)
            _ = shutil.copy(curr_img_path, curr_save_path)

        report = classification_report(y_list,pred_list)
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