from work.stem_extraction.stem_extract import *

from work.auxiliary.logger_settings import configure_logger
from work.auxiliary import data_functions
import logging

LOG_PATH = os.path.abspath('logs')
DATA_PATH = os.path.abspath('data')

log_path = data_functions.create_path(LOG_PATH, 'stem_extract')

configure_logger(name="stem_extract",
                 console_level='INFO',
                 file_level='INFO',
                 out_path=log_path)

logger = logging.getLogger(__name__)


def main():
    dest_path = os.path.join(DATA_PATH, r'annotation_output\New_folder')

    img_path = os.path.join(DATA_PATH, r'raw_data\with_maskes\image')
    # img_path = os.path.join(DATA_PATH, r'raw_data\images_orig')

    # mask_path = os.path.join(dest_path, 'raw_pred')
    mask_path = os.path.join(DATA_PATH, r'raw_data\with_maskes\label')
    mask_path = os.path.join(DATA_PATH,r'unet_data\training\2019-09-30_07-19-46\raw_pred')

    ground_path = os.path.join(DATA_PATH, r'classification_data\from_all\set1')

    threshold = 0.4
    is_binary_mask = False
    hist_type = 'bgr'
    object_type = 'ontop'

    save_flag = True
    create_ground_obj = False

    # create ground truth
    if create_ground_obj:
        data_functions.create_raw_test_train_ground_truth(ground_path,
                                                          mask_path,
                                                          dest_path)

    create_test_train_obj(ground_path=ground_path,
                          mask_path=mask_path,
                          save_path=dest_path,
                          threshold=threshold,
                          hist_type=hist_type,
                          use_thres_flag=True,
                          obj_type=object_type,
                          is_binary_mask=is_binary_mask)

    create_object(img_path=img_path,
                  mask_path=mask_path,
                  save_path=dest_path,
                  threshold=threshold,
                  hist_type=hist_type,
                  use_thres_flag=False,
                  obj_type=object_type,
                  is_binary_mask=is_binary_mask)


if __name__ == '__main__':
    main()
