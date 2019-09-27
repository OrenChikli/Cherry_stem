from work.stem_extraction.stem_extract import *

from logger_settings import *
configure_logger()
logger = logging.getLogger("extraction_main")


def get_pred_histograms(img_path, mask_path, save_path, threshold):

    logger.info(f"getting ground truth with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    src_path=save_path,
                                    threshold=threshold)
    logger.info("saving ground truth histogrames")
    stem_exctractor.calc_hists()

def compare_hists(img_path,mask_path,save_path,threshold,ground_truth_path):
    logger.info(f"intializing StemExtractor instance with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    src_path=save_path,
                                    threshold=threshold)
    logger.info("loading groung truth histograms")
    stem_exctractor.load_ground_truth(ground_truth_path)

    logger.info("comparing ground truth histogrames with predictions")
    stem_exctractor.compare_hists()





def main():

    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    ground_truth_path = r'D:\Clarifruit\cherry_stem\data\raw_data\stem classes\thres_100\hist_orig'

    src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-27_16-15-36'
    masks_folder='raw_pred'
    mask_path = os.path.join(src_path,masks_folder)

    threshold = 150


    #data_functions.get_masks_via_img(img_path,src_mask_path,mask_path)
    #get_pred_histograms(img_path, src_mask_path, save_path, threshold)

    general_use(img_path, mask_path,src_path, threshold)

    #compare_hists(img_path, src_mask_path, save_path, threshold, ground_truth_path)


def general_use(img_path, mask_path, src_path, threshold):
    logger.info(f"intializing StemExtractor instance with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    src_path=src_path,
                                    threshold=threshold)
    logger.info("saving binary_masks")
    x=stem_exctractor.raw_mask_loader()
    for i in x:
        print(i)
    #stem_exctractor.get_threshold_masks()
    # logger.info("getting sharpened masks")
    # stem_exctractor.sharpen_maskes()
    # logger.info("saving sharpened binary masks")
    # stem_exctractor.get_threshold_masks(type_flag='sharp')
    # # ----------------------------------------------------------
    # logger.info(f"getting stems from {type_flag} masks")
    # stem_exctractor.get_stems(type_flag=type_flag)
    # logger.info(f"getting stems mean color from {type_flag} masks")
    # stem_exctractor.get_mean_color(type_flag=type_flag)
    # logger.info(f"getting stems mean heu color from {type_flag} masks")
    # stem_exctractor.get_mean_h(type_flag=type_flag)
    # logger.info(f"getting mean heu stem scors for {type_flag} masks")
    # stem_exctractor.score_results(score_type='heu', mask_type=type_flag)
    # logger.info(f"getting mean color stem scors for {type_flag} masks")
    # stem_exctractor.score_results(score_type='color', mask_type=type_flag)


if __name__ == '__main__':
    main()
