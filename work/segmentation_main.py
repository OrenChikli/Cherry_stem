from work.auxiliary import data_functions
import cv2

from work.segmentation import segmentation, seg_finder_with_ground_truth
import os

from work.auxiliary import logger_settings

REAL_PATH = os.path.abspath('..')
LOG_PATH = os.path.join(REAL_PATH, 'logs')
DATA_PATH = os.path.join(REAL_PATH, 'data')


log_path = data_functions.create_path(LOG_PATH, 'segmentation_logs')
logger = logger_settings.configure_logger(name="segmentation",
                          console_level='INFO',
                          file_level='DEBUG',
                          out_path=log_path)


def use_seg_finder_with_ground_truth(img_path, mask_path):
    sf = seg_finder_with_ground_truth.MaskSegmentFinder(img_path, mask_path)

    sf.display()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def seg_single():
    #img_name = '74714-32897.png.jpg'
    # image_name = '45665-81662.png.jpg'
    img_name = '38360-25986.png.jpg'

    # orig_path =r'D:\Clarifruit\cherry_stem\data\difficult\image'
    orig_path = os.path.join(DATA_PATH,r'raw_data\with_maskes\image')
    mask_path = os.path.join(DATA_PATH,r'raw_data\with_maskes\label')
    seg_path = os.path.join(DATA_PATH,r'segmentation')

    img_path = os.path.join(orig_path, img_name)
    mask_path = os.path.join(mask_path, img_name)

    display_flag= True
    save_flag=True
    save_segments = False

    # settings_dict = {'pr_threshold': 0.15,
    #                  'seg_type':"slic",
    #                  'seg_params': dict(n_segments=3000,max_iter=200, sigma=0.5,
    #                                     compactness=10.0,
    #                                     enforce_connectivity=True,
    #                                     min_size_factor=0.05,
    #                                     max_size_factor=3),
    #                  'gray_scale': False}

    # settings_dict = {'pr_threshold': 0.2,
    #                  'seg_type':"slic",
    #                  'seg_params': dict(n_segments=2000,max_iter=50, sigma=0.0,
    #                                     compactness=20.0,
    #                                     enforce_connectivity=True,
    #                                     min_size_factor=0.1,
    #                                     max_size_factor=2,
    #                                     convert2lab=True,
    #                                     slic_zero=False),
    #                  'gray_scale': False}




    settings_dict = {'pr_threshold': 0.15,
                     'seg_type':"felzenszwalb",
                     'seg_params': {'scale': 1, 'sigma': 0,'min_size': 5},
                     'gray_scale': False}

    sg = segmentation.SegmentationSingle(img_path=img_path,
                                         mask_path=mask_path,
                                         is_binary_mask=True,
                                         save_path=seg_path,
                                         create_save_dest_flag=True,
                                         **settings_dict)

    sg.apply_segmentation(save_flag=save_flag, display_flag=display_flag,
                          save_segments=save_segments)
    sg.get_ontop(display_flag=display_flag,save_flag=save_flag)
    cv2.waitKey(0)



def main():
    image_name = 'orig_72596-28736.png.jpg'

    #img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes\image'
    orig_path = os.path.join(DATA_PATH,r'raw_data\with_maskes\image')
    mask_path = os.path.join(DATA_PATH,r'raw_data\with_maskes\label')
    seg_path = os.path.join(DATA_PATH,r'segmentation')
    #mask_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46\raw_pred'

    is_binary_mask=True

    img_list = [image_name]
    # img_list = [
    #     '38360-00777.png.jpg',
    #     '38360-02397.png.jpg',
    #     '38360-25986.png.jpg',
    #     '38360-27560.png.jpg',
    #     '38360-46226.png.jpg',
    #     '38360-68930.png.jpg',
    # ]


    settings_dict = {'pr_threshold': 0.5,
                     'seg_type':"quickshift",
                     'seg_params': {},
                     'gray_scale': False}

    multi_sg = segmentation.SegmentationMulti(img_path=img_path, mask_path=mask_path,
                                              seg_path=seg_path,is_binary_mask=is_binary_mask)
    multi_sg.segment_multi(settings_dict, img_list=None)


if __name__ == '__main__':
    #main()
    seg_single()
