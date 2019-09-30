from work.stem_classifier import classify
import os

from logger_settings import *
configure_logger()
logger = logging.getLogger("classifier_main")


def main():
    train_path= r'D:\Clarifruit\cherry_stem\data\classification_data\normal_classification\ground'


    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'

    test_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46'

    threshold = 0.4


    classifier = classify.StemHistClassifier(train_path=train_path,
                                             test_path=test_path,
                                             threshold=threshold)

    model_name = 'LogisticRegression'
    model_parameters = {'solver': 'lbfgs',
                        'multi_class': 'auto',
                        'max_iter': 1000}

    classifier.train_model(model_name, **model_parameters)
    classifier.model_predict(img_path)



if __name__ == '__main__':
    main()

