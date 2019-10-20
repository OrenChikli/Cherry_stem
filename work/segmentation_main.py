from work.auxiliary import data_functions
import cv2

from work.segmentation import segmentation
import os

from work.auxiliary.logger_settings import configure_logger
import logging

LOG_PATH = os.path.abspath('logs')
DATA_PATH = os.path.abspath('data')

log_path = data_functions.create_path(LOG_PATH, 'segmentation_logs')

configure_logger(name="segmentation",
                 console_level='INFO',
                 file_level='INFO',
                 out_path=log_path)

logger = logging.getLogger(__name__)


def main():
    orig_path = os.path.join(DATA_PATH, r'raw_data\with_maskes\image')
    # mask_path = os.path.join(DATA_PATH,r'raw_data\with_maskes\label')
    mask_path = os.path.join(DATA_PATH,
                             r'unet_data\training\2019-10-20_19-26-14\raw_pred')
    dest_path = os.path.join(DATA_PATH,
                             r'unet_data\training\2019-10-20_19-26-14')
    # dest_path = os.path.join(DATA_PATH,r'segmentation')

    is_binary_mask = False

    single_flag = False  # segment single image or multiple

    ## setings for single
    img_name = '38360-25986.png.jpg'

    display_flag = True
    save_flag = True
    save_segments = False

    # settings for multi segmentation
    img_list = None

    # img_list = [
    #     '38360-00777.png.jpg',
    #     '38360-02397.png.jpg',
    #     '38360-25986.png.jpg',
    #     '38360-27560.png.jpg',
    #     '38360-46226.png.jpg',
    #     '38360-68930.png.jpg',
    # ]

    # general settings for segmentation
    settings_dict = {'threshold': 0.4,
                     "pr_threshold": 0.5,
                     'seg_type': "quickshift",
                     'seg_params': {},
                     'gray_scale': False}

    if single_flag:
        img_path = os.path.join(orig_path, img_name)
        mask_path = os.path.join(mask_path, img_name)

        sg = segmentation.SegmentationSingle(img_path=img_path,
                                             mask_path=mask_path,
                                             is_binary_mask=is_binary_mask,
                                             save_path=dest_path,
                                             create_save_dest_flag=True,
                                             **settings_dict)

        sg.apply_segmentation(save_flag=save_flag, display_flag=display_flag,
                              save_segments=save_segments)
        sg.get_ontop(display_flag=display_flag, save_flag=save_flag)
        if display_flag:
            cv2.waitKey(0)

    else:

        segmentation.segment_multi(img_path=orig_path,
                                   mask_path=mask_path,
                                   save_path=dest_path,
                                   is_binary_mask=is_binary_mask,
                                   settings_dict=settings_dict,
                                   img_list=img_list)


if __name__ == '__main__':
    main()
