from logger_settings import configure_logger
import logging
from work.unet_main import load_from_files
configure_logger()
logger = logging.getLogger('full_model')


def main():
    train_path=r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes'
    dest_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training'
    test_path=r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    src_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46'

    model = load_from_files(src_path)
    for img_name,pred in model.prediction_generator(src_path):
        pass



if __name__ == '__main__':
    main()