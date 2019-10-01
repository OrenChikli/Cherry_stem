from logger_settings import configure_logger
from auxiliary.custom_image import CustomImage
import logging
from work.unet_main import load_from_files
configure_logger()
logger = logging.getLogger('full_model')


def main():
    train_path=r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes'
    dest_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training'
    test_path=r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    src_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46'
    hist_type = 'bgr'
    threshold=0.4
    model = load_from_files(src_path)
    for img_entry,pred in model.prediction_generator(test_path):
        curr_image = CustomImage(img_path=img_entry.path,threshold=threshold,mask=pred)
        hist= curr_image.get_hist_via_mask(hist_type=hist_type)
        print("g")



if __name__ == '__main__':
    main()