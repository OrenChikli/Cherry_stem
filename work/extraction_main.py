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
    src_path = os.path.join(DATA_PATH,
                            r'unet_data\training\2019-10-20_19-26-14')
    img_path = os.path.join(DATA_PATH, r'raw_data\images_orig')

    mask_path = os.path.join(src_path, 'raw_pred')

    ground_path = os.path.join(DATA_PATH, r'classification_data\from_all\set1')

    threshold = 0.4
    hist_type = 'bgr'
    object_type = 'ontop'

    save_flag = True

    # create ground truth
    ground_train_path = os.path.join(src_path, 'train')
    if not os.path.exists(ground_train_path):
        data_functions.create_raw_test_train_ground_truth(ground_path,
                                                          mask_path,
                                                          src_path)

    create_test_train_obj(ground_path=ground_path,
                          threshold=threshold,
                          src_path=src_path,
                          hist_type=hist_type,
                          obj_type=object_type)

    create_object(img_path=img_path,
                  mask_path=mask_path,
                  save_path=src_path,
                  threshold=threshold,
                  hist_type=hist_type,
                  use_thres_flag=True,
                  obj_type=object_type)


if __name__ == '__main__':
    main()
