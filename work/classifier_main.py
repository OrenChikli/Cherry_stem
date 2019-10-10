from work.stem_classifier import classify
from auxiliary import data_functions
import os
import xgboost as xgb

from logger_settings import *
configure_logger()
logger = logging.getLogger("classifier_main")




def main():

    src_path= r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-10-07_20-12-39\thres_0.4'
    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'

    train_folder=r'train'
    test_folder = 'test'
    train_path = os.path.join(src_path,train_folder)
    test_path = os.path.join(src_path, test_folder)

    hist_type = 'hsv'
    #test_folder = f'{hist_type}_histograms'
    #test_path = os.path.join(src_path,test_folder)

    save_folder ='classification_preds'
    save_path = data_functions.create_path(src_path,save_folder)

    classifier = classify.StemHistClassifier(train_path=train_path,
                                             hist_type=hist_type)

    model_name = 'LogisticRegression'
    model_parameters = {'solver': 'lbfgs',
                        'multi_class': 'multinomial',
                        'class_weight' : 'balanced',
                        'max_iter': 5000}

    model_name = 'RandomForestClassifier'
    #model_parameters ={'n_estimators':1000}

    model_name ='Xgboost'
    model_parameters = {}

    #classifier.train_model(save_path, model_name)
    classifier.train_model(save_path,model_name, **model_parameters)

    classifier.model_predict(test_path,save_path,img_path)



if __name__ == '__main__':
    #get_test_train()
    main()

