from work.stem_classifier.classifier import classify
from work.logger_settings import *

configure_logger()
logger = logging.getLogger(__name__)


def main():
    train_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\ground'
    test_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_23-55-20\masks\thres_150\hist_orig'
    dest_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_23-55-20\masks\thres_150\classifier_scores'
    src_images_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    trained_threshold = 150

    classifier = classify.StemHistClassifier(train_path=train_path,
                                             test_path=test_path,
                                             save_path=dest_path,
                                             created_thres=trained_threshold)

    model_name = 'LogisticRegression'
    model_parameters = {'solver': 'lbfgs',
                        'multi_class': 'auto',
                        'max_iter': 1000}

    classifier.train_model(model_name, **model_parameters)
    classifier.model_predict(src_images_path)



if __name__ == '__main__':
    main()

