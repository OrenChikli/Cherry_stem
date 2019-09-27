from work.auxiliary.data_functions import *

from work.segmentation.clarifruit_segmentation import segmentation, seg_finder_with_ground_truth

from datetime import datetime
from work.auxiliary.display_functions import *
from work.logger_settings import *
import logging


configure_logger()
logger = logging.getLogger(__name__)

def segment_multi(orig_path, mask_path, seg_path,settings_dict,img_list=None):
    logger.debug(" <- segment_multi")
    if img_list is None:
        img_list = [img_entry.name for img_entry in os.scandir(orig_path)]

    dir_save_path = create_path(seg_path, 'several')

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    dir_save_path = create_path(dir_save_path, current_time)
    save_json(settings_dict, "segmentation_settings.json", dir_save_path)
    logger.info(f"segmenting to {dir_save_path}")

    for img in img_list:
        curr_img_path = os.path.join(orig_path,img)
        curr_mask_path = os.path.join(mask_path,img)
        curr_segment = segmentation.Segmentation(img_path=curr_img_path,mask_path=curr_mask_path,
                                                  scale=settings_dict['scale'],
                                                  sigma=settings_dict['sigma'],
                                                  min_size=settings_dict['min_size'],
                                                  pr_threshold=settings_dict['pr_threshold'])

        res_mask = curr_segment.return_modified_mask()

        base_name = img.rsplit('.',1)[0]
        save_path_mask = os.path.join(dir_save_path,f"{base_name}_mask.jpg")
        cv2.imwrite(save_path_mask,res_mask)

    logger.debug(" -> segment_multi")




def use_segment(image_name,orig_path,mask_path,seg_path,settings_dict):

    seg_folder = 'segmentation'
    seg_activation_folder = 'activation'

    img_path = os.path.join(orig_path,image_name)
    img_mask_path = os.path.join(mask_path,image_name)

    #TODO insert new segmentation class




def use_seg_finder_with_ground_truth(img_path,mask_path):
    sf = seg_finder_with_ground_truth.MaskSegmentFinder(img_path,mask_path)

    sf.display()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def new_segmentation():
    image_name = '72596-28736.png.jpg'

    #orig_path =r'D:\Clarifruit\cherry_stem\data\difficult\image'
    orig_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes\image'
    mask_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes\label'

    img_path = os.path.join(orig_path,image_name)
    mask_path =os.path.join(mask_path,image_name)
    settings_dict = {'pr_threshold': 0.3,
                     'scale': 100,
                     'sigma': 0.1,
                     'min_size': 60}

    sg = segmentation.Segmentation(img_path,mask_path,**settings_dict)
    sg.apply_segmentation()
    res = put_binary_ontop(sg.img,sg.filtered_segments)
    plt.imshow(res)
    plt.show()


def seg_main():
    logger.debug(" <- seg main")
    image_name = 'orig_72596-28736.png.jpg'

    #orig_path =r'D:\Clarifruit\cherry_stem\data\difficult\image'
    orig_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes\image'
    mask_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\binary_thres_0.5'
    seg_path =  r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_00-55-09\segmentation'

    seg_folder='segmentation'
    seg_activation_folder ='activation'

    img_path = os.path.join(orig_path, image_name)
    img_mask_path = os.path.join(mask_path,image_name)

    img_list = [
                '45783-98635.png.jpg',
                '71089-01084.png.jpg',
                '72492-85602.png.jpg',
                '72520-70104.png.jpg',
                '72590-54586.png.jpg',
                '72592-11978.png.jpg',
                '72596-28736.png.jpg',
                '74714-32897.png.jpg',
                '78702-22132.png.jpg',
                '78702-32898.png.jpg',
                '78702-35309.png.jpg',
                '78712-02020.png.jpg']

    settings_dict = {'threshold': 1,
                     'pr_threshold': 0.3,
                     'scale': 50,
                     'sigma': 0.1,
                     'min_size': 20}

    #use_segment(image_name,orig_path, mask_path, seg_path, settings_dict)
    #use_seg_info(img_path)
    #use_seg_finder(img_path)
    #use_seg_filter(img_path)
    #use_seg_finder_with_ground_truth(img_path,img_mask_path)
    #img_list = [item.name for item in os.scandir(orig_path)]
    segment_multi(orig_path, mask_path, seg_path,settings_dict)
    logger.debug(" -> seg_main")

if __name__ == '__main__':


    seg_main()
    #new_segmentation()