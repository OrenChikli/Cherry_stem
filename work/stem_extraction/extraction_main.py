from work.preprocess import data_functions
from work.stem_extraction.stem_extract import *

from work.logger_settings import *

configure_logger()
logger = logging.getLogger(__name__)

from work.segmentation.clarifruit_segmentation.image import Image

def get_ground_truth():

    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\stems'

    mask_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\label'
    save_path=r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes'

    threshold = 100

    logger.info(f"getting ground truth with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    save_path=save_path,
                                    threshold=threshold)
    logger.info("saving ground truth histogrames")
    stem_exctractor.calc_hists()

def compare_hists(img_path,mask_path,save_path,threshold,ground_truth_path):
    logger.info(f"intializing StemExtractor instance with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    save_path=save_path,
                                    threshold=threshold)
    logger.info("loading groung truth histogrames")
    stem_exctractor.load_ground_truth(ground_truth_path)

    logger.info("comparing ground truth histogrames with predictions")
    stem_exctractor.compare_hists()




def main():
    h_classes_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\hsv'
    color_classes_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\colors'

    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'


    mask_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_23-55-20\masks\raw_masks'

    save_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-22_23-55-20\masks'

    ground_truth_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\thres_100\hist_orig'

    threshold = 100

    type_flag = 'orig'

    #general_use(color_classes_path, h_classes_path, img_path, mask_path, threshold, type_flag)

    compare_hists(img_path, mask_path, save_path, threshold, ground_truth_path)


def general_use(color_classes_path, h_classes_path, img_path, mask_path,save_path, threshold, type_flag):
    logger.info(f"intializing StemExtractor instance with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    save_path=save_path,
                                    h_clasess_path=h_classes_path,
                                    color_classes_path=color_classes_path,
                                    threshold=threshold)
    logger.info("saving binary_masks")
    stem_exctractor.get_threshold_masks()
    logger.info("getting sharpened masks")
    stem_exctractor.sharpen_maskes()
    logger.info("saving sharpened binary masks")
    stem_exctractor.get_threshold_masks(type_flag='sharp')
    # ----------------------------------------------------------
    logger.info(f"getting stems from {type_flag} masks")
    stem_exctractor.get_stems(type_flag=type_flag)
    logger.info(f"getting stems mean color from {type_flag} masks")
    stem_exctractor.get_mean_color(type_flag=type_flag)
    logger.info(f"getting stems mean heu color from {type_flag} masks")
    stem_exctractor.get_mean_h(type_flag=type_flag)
    logger.info(f"getting mean heu stem scors for {type_flag} masks")
    stem_exctractor.score_results(score_type='heu', mask_type=type_flag)
    logger.info(f"getting mean color stem scors for {type_flag} masks")
    stem_exctractor.score_results(score_type='color', mask_type=type_flag)


if __name__ == '__main__':
    main()
    #get_ground_truth()