from work.stem_extraction.stem_extract import *

from work.auxiliary.logger_settings import configure_logger
from work.auxiliary import data_functions
import logging

FUNC_DICTIPNARY = {'binary': lambda x: x.get_threshold_masks(),
                   'hists': lambda x: x.calc_hists(),
                   'ontop': lambda x: x.ontop(),
                   'stems': lambda x: x.get_stems(),
                   'filter_images': lambda x: x.filter_images()}

LOG_PATH = os.path.abspath('logs')
DATA_PATH = os.path.abspath('data')

log_path = data_functions.create_path(LOG_PATH, 'stem_extract')

configure_logger(name="stem_extract",
                 console_level='INFO',
                 file_level='INFO',
                 out_path=log_path)

logger = logging.getLogger(__name__)


def create_object(img_path, mask_path, save_path, threshold, hist_type,
                  use_thres_flag,
                  obj_type):
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    save_path=save_path,
                                    threshold=threshold,
                                    use_thres_flag=use_thres_flag,
                                    hist_type=hist_type)

    FUNC_DICTIPNARY[obj_type](stem_exctractor)


def create_raw_ground_truth(img_path, mask_path, dest_path,
                            train_folder='classifier_train_data'):
    """
    getting respective masks of images in diffrent classes.
    a method to get the predictions from src_mask_path, into the y_folder in
    the ground_path, where the x_folder has the source images, selected by the
    user
    :param img_path: a path containing the ground truth, has a structure of
    x_folder with src images, y_folder with labels
    :param mask_path: the image where all the available predictions reside
    """
    curr_dest_path = data_functions.create_path(dest_path, train_folder)
    for curr_class in os.scandir(img_path):
        curr_dest = data_functions.create_path(curr_dest_path, curr_class.name)

        logger.info(f"getting ground truth for class {curr_class.name}")
        logger.info(f"copying ground truth from {curr_class.path}")
        logger.info(f"copying ground truth to {curr_dest}")

        data_functions.get_masks_via_img(curr_class.path, mask_path, curr_dest)


def create_ground_truth_objects(ground_path, threshold, src_path, obj_type,
                                train_folder='classifier_train_data',
                                hist_type='bgr'):
    dest_path = data_functions.create_path(src_path, f"thres_{threshold}")
    dest_path = data_functions.create_path(dest_path, train_folder)
    raw_pred_path = os.path.join(src_path, train_folder)

    for curr_class in os.scandir(raw_pred_path):
        curr_raw_pred_path = os.path.join(raw_pred_path, curr_class.name)
        curr_dest = data_functions.create_path(dest_path, curr_class.name)
        curr_ground_path = os.path.join(ground_path, curr_class.name)

        create_object(img_path=curr_ground_path,
                      mask_path=curr_raw_pred_path,
                      save_path=curr_dest,
                      threshold=threshold,
                      hist_type=hist_type,
                      use_thres_flag=False,
                      obj_type=obj_type)


def get_test_train():
    """
    use the src_path to create a test train split of data seperated to clases,
    where each class has it's own folder.
    save the test and train results in dest_path
    :return:
    """
    src_path = r'D:\Clarifruit\cherry_stem\data\classification_data\from_all\set3\All'
    dest_path = r'D:\Clarifruit\cherry_stem\data\classification_data\from_all\set3'
    data_functions.get_train_test_split(src_path, dest_path,
                                        train_name='train',
                                        test_name='test',
                                        test_size=0.33)


def main():
    src_path = os.path.join(DATA_PATH,
                            r'unet_data\training\2019-10-19_21-18-37')
    img_path = os.path.join(DATA_PATH, r'raw_data\images_orig')

    mask_path = os.path.join(src_path, 'raw_pred')

    ground_path = os.path.join(DATA_PATH, r'classification_data\from_all\set1')
    ground_train_path = os.path.join(ground_path, 'train')
    ground_test_path = os.path.join(ground_path, 'test')

    threshold = 0.4
    hist_type = 'bgr'
    object_type = 'hists'

    save_flag = True

    # create ground truth
    # create train data
    # create_raw_ground_truth(ground_train_path, mask_path, src_path,
    #                     train_folder='train')
    # # create test data
    # create_raw_ground_truth(ground_test_path, mask_path, src_path,
    #                         train_folder='test')


    create_ground_truth_objects(ground_path=ground_train_path,
                                threshold=threshold,
                                src_path=src_path,
                                train_folder='train',
                                hist_type=hist_type,
                                obj_type=object_type)


    create_ground_truth_objects(ground_path=ground_test_path,
                                threshold=threshold,
                                src_path=src_path,
                                train_folder='test',
                                hist_type=hist_type,
                                obj_type=object_type)
    # experiment with current data

    # create_object(img_path, mask_path, save_path, threshold, hist_type,
    #               use_thres_flag,
    #               obj_type)
    # get_binary_masks(img_path, mask_path, src_path, threshold)
    # # create_stems(img_path,mask_path,src_path,threshold)
    # ontop(img_path, mask_path, src_path, threshold)
    # # filtter_images(img_path, mask_path, src_path, threshold, save_flag)
    # # get_pred_histograms(img_path, mask_path, src_path, threshold, hist_type)


if __name__ == '__main__':
    # get_test_train()
    main()
