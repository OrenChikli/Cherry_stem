from work.preprocess import data_functions
from work.stem_extraction.stem_extract import *

from work.logger_settings import *

configure_logger()
logger = logging.getLogger(__name__)

from work.segmentation.clarifruit_segmentation.image import Image



def main():
    h_classes_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\hsv'
    color_classes_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\colors'

    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'


    mask_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_23-55-20\masks'

    threshold = 100

    type_flag = 'orig'

    logger.info(f"intializing StemExtractor instance with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    h_clasess_path=h_classes_path,
                                    color_classes_path=color_classes_path,
                                    threshold=threshold)
    logger.info("saving binary_masks")
    stem_exctractor.get_threshold_masks()

    logger.info("getting sharpened masks")
    stem_exctractor.sharpen_maskes()

    logger.info("saving sharpened binary masks")
    stem_exctractor.get_threshold_masks(type_flag='sharp')

    #----------------------------------------------------------

    logger.info(f"getting stems from {type_flag} masks")
    stem_exctractor.get_stems(type_flag=type_flag)

    logger.info(f"getting stems mean color from {type_flag} masks")
    stem_exctractor.get_mean_color(type_flag=type_flag)

    logger.info(f"getting stems mean heu color from {type_flag} masks")
    stem_exctractor.get_mean_h(type_flag=type_flag)

    logger.info(f"getting mean heu stem scors for {type_flag} masks")
    stem_exctractor.score_results(score_type='heu',mask_type=type_flag)

    logger.info(f"getting mean color stem scors for {type_flag} masks")
    stem_exctractor.score_results(score_type='color',mask_type=type_flag)

if __name__ == '__main__':
    main()