from work.stem_classifier import classify
from auxiliary import data_functions
import os

from logger_settings import *
configure_logger()
logger = logging.getLogger("classifier_main")

def get_test_train():
    src_path=r'D:\Clarifruit\cherry_stem\data\classification_data\from_all\All'
    dest_path =r'D:\Clarifruit\cherry_stem\data\classification_data\from_all'
    data_functions.get_train_test_split(src_path,dest_path,train_name='train',test_name='test',test_size=0.33)

def main():

    src_path= r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46\thres_0.4'
    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'

    train_folder=r'train'
    train_path = os.path.join(src_path,train_folder)

    hist_type = 'hsv'
    test_folder = f'{hist_type}_histograms'
    test_path = os.path.join(src_path,test_folder)

    save_folder ='classification_preds'
    save_path = data_functions.create_path(src_path,save_folder)

    classifier = classify.StemHistClassifier(train_path=train_path,
                                             hist_type=hist_type)

    model_name = 'LogisticRegression'
    model_parameters = {'solver': 'lbfgs',
                        'multi_class': 'auto',
                        'max_iter': 1000}

    classifier.train_model(save_path,model_name, **model_parameters)

    classifier.model_predict(test_path,save_path,img_path)



if __name__ == '__main__':
    #get_test_train()
    main()

