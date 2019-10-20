from work.auxiliary.logger_settings import configure_logger
from work.auxiliary import data_functions
import os
import logging


LOG_PATH = os.path.abspath('logs')
DATA_PATH = os.path.abspath('data')

log_path = data_functions.create_path(LOG_PATH, 'stem_extract')

configure_logger(name="stem_extract",
                 console_level='INFO',
                 file_level='INFO',
                 out_path=log_path)

logger = logging.getLogger(__name__)

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