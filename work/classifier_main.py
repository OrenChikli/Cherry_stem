from work.stem_classifier import classify
from work.auxiliary import data_functions
import os
from work.stem_classifier.model_functions import *

from work.auxiliary.logger_settings import configure_logger
LOG_PATH = os.path.abspath('logs')
DATA_PATH = os.path.abspath('data')

log_path = data_functions.create_path(LOG_PATH, 'classifier')

configure_logger(name="classifier",
                 console_level='INFO',
                 file_level='INFO',
                 out_path=log_path)

logger = logging.getLogger(__name__)


def func():
    train_path=r'D:\Clarifruit\cherry_stem\data\classification_data\from_all\set1\train'
    dest_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-10-07_20-12-39'
    test_path=r'D:\Clarifruit\cherry_stem\data\classification_data\from_all\set1\test'

    params_dict = dict(

        train_path=train_path,

        data_gen_args=dict(rescale=1. / 255,
                           rotation_range=180,
                           width_shift_range=0.25,
                           height_shift_range=0.25,
                           shear_range=0.2,
                           zoom_range=[0.5, 1.0],
                           horizontal_flip=True,
                           vertical_flip=True,
                           fill_mode='nearest'),

        optimizer='Adam',
        optimizer_params=dict(lr=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        pretrained_weights=None,

        target_size=(256, 256),
        color_mode='rgb',
        batch_size=10,
        epochs=10,
        steps_per_epoch=3000,
        valdiation_split=0.2,
        validation_steps=300)


    logger.info("created training instance")
    model = ClarifruitClassifier(**params_dict)
    logger.info("train start")
    model.train_model(dest_path=dest_path,params_dict=params_dict)
    logger.info("finished training")

    model.prediction(test_path,dest_path)


def main():
    src_path = os.path.join(DATA_PATH,
                            r'unet_data\training\2019-10-20_19-26-14\thres_0.4')
    img_path = os.path.join(DATA_PATH, r'raw_data\images_orig')

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
                                             hist_type=hist_type,
                                             label_col='label',
                                             drop_cols='file_name')

    model_name = 'LogisticRegression'
    model_parameters = {'solver': 'lbfgs',
                        'multi_class': 'multinomial',
                        'class_weight' : 'balanced',
                        'max_iter': 5000}


    model_name ='XGBClassifier'
    model_parameters = {'colsample_bytree': 0.5,
                        'eta': 0.03,
                        'eval_metric': 'mlogloss',
                        'max_depth': 7,
                        'nthread': 2,
                        'num_class': 4,
                        'num_round': 1000,
                        'objective': 'multi:softmax',
                        'silent': 1,
                        'subsample': 0.4,
                        'n_estimators': 1000}

    model_parameters = {'colsample_bytree': 0.5,
                        'eta': 0.03,
                        'max_depth': 7,
                        'nthread': 2,
                        'num_round': 1000,
                        'objective': 'binary:logistic',
                        'silent': 1,
                        'subsample': 0.4,
                        'n_estimators': 1000}


    #classifier.train_model(save_path, model_name)
    classifier.train_model(save_path,model_name, **model_parameters)

    classifier.model_predict(test_path,save_path,img_path)



if __name__ == '__main__':
    #get_test_train()
    main()
    #func()

