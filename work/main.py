from logger_settings import configure_logger

from auxiliary import data_functions
from work.complete_model.model import TrainedModel
import logging
import shutil


configure_logger()
logger = logging.getLogger('full_model')

def save_results(res,save_path):
    for img_entry,pred in res:
        curr_save_path = data_functions.create_path(save_path, pred)
        _ = shutil.copy(img_entry.path, curr_save_path)


def main():

    test_path=r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'

    unet_model_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46'
    classifier_path = unet_model_path+r'\thres_0.4\classification_preds\2019-10-01_19-52-55\saved_model'
    save_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46\thres_0.4\classification_preds\New folder'
    model = TrainedModel(unet_model_path=unet_model_path,classifier_path=classifier_path)
    res = model.predict(test_path)
    save_results(res,save_path)


if __name__ == '__main__':
    main()