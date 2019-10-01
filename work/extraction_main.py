from work.stem_extraction.stem_extract import *

from logger_settings import *
configure_logger()
logger = logging.getLogger("extraction_main")

def get_binary_masks(img_path, mask_path, src_path, threshold):
    """
    create binary masks via the given threshold of the makes given in the mask_path and save them in the src_path
    :param img_path:
    :param mask_path:
    :param src_path:
    :param threshold:
    :return:
    """
    logger.info(f"intializing StemExtractor instance with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    src_path=src_path,
                                    threshold=threshold)
    logger.info("saving binary_masks")
    stem_exctractor.get_threshold_masks()


def create_stems(img_path,mask_path,save_path,threshold):
    logger.info(f"intializing StemExtractor instance with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    src_path=save_path,
                                    threshold=threshold)
    logger.info("getting stems")
    stem_exctractor.get_stems()

def ontop(img_path,mask_path,save_path,threshold):
    logger.info(f"intializing StemExtractor instance with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    src_path=save_path,
                                    threshold=threshold)
    logger.info("getting ontop images")
    stem_exctractor.get_ontop_images()

def filtter_images(img_path, mask_path, save_path, threshold,lower,upper):
    logger.info(f"intializing StemExtractor instance with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    src_path=save_path,
                                    threshold=threshold)
    logger.info("getting filltered images")
    stem_exctractor.fillter_via_color(lower, upper)

def get_pred_histograms(img_path, mask_path, save_path, threshold,hist_type='bgr',use_thres_flag=True):

    logger.info(f"getting {hist_type} histograms with threshold: {threshold}")
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    src_path=save_path,
                                    threshold=threshold,
                                    use_thres_flag=use_thres_flag)
    logger.info("saving histograms")
    stem_exctractor.calc_hists(hist_type=hist_type)



def create_ground_truth(img_path, mask_path,dest_path,train_folder='classifier_train_data'):
    """
    getting respective masks of images in diffrent classes.
    a method to get the predictions from src_mask_path, into the y_folder in the ground_path, where the x_folder has
    the source images, selected by the user
    :param img_path: a path containig the ground truth, has structure of x_folder with src images, y_folder with labels
    :param src_mask_path: the image where all the avaliabel predictions reside
    :param x_folder: the name of the subfolder in ground truth which has the src images
    :param y_folder: the name of the subfolder in the ground truth path which will house the labels
    :return: None
    """
    curr_dest_path = data_functions.create_path(dest_path,train_folder)
    for curr_class in os.scandir(img_path):
        logger.info(f"getting masks for path {curr_class.path}")
        curr_dest = data_functions.create_path(curr_dest_path,curr_class.name)
        data_functions.get_masks_via_img(curr_class.path,mask_path,curr_dest)





def create_ground_truth_hists(ground_path,threshold,src_path,
                              train_folder='classifier_train_data',
                              hist_type='bgr'):

    dest_path = data_functions.create_path(src_path,f"thres_{threshold}")
    dest_path = data_functions.create_path(dest_path, train_folder)
    raw_pred_path = os.path.join(src_path, train_folder)

    for curr_class in os.scandir(raw_pred_path):
        curr_raw_pred_path = os.path.join(raw_pred_path,curr_class.name)
        logger.info(f"getting masks for path {curr_raw_pred_path}")
        curr_dest = data_functions.create_path(dest_path, curr_class.name)
        logger.info(f"saving {hist_type} histograms at {curr_dest}")
        curr_ground_path =os.path.join(ground_path,curr_class.name)
        get_pred_histograms(curr_ground_path,
                            curr_raw_pred_path,
                            curr_dest,
                            threshold,
                            hist_type,
                            use_thres_flag=False)





def main():
    src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46'

    train_folder = 'classifier_train_data'
    raw_preds_folder = 'raw_pred'
    mask_path = os.path.join(src_path,raw_preds_folder)
    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    ground_path =r'D:\Clarifruit\cherry_stem\data\classification_data\normal_classification'


    threshold = 0.4
    hist_type='bgr'

    lower=(0,0,0)
    upper = (230,255,230)

    # create train data
    # create_ground_truth(ground_path, mask_path, src_path,
    #                     train_folder=train_folder)
    create_ground_truth_hists(ground_path=ground_path,threshold= threshold,src_path= src_path,
                              train_folder=train_folder,hist_type=hist_type)

    #experiment with current data

    #get_binary_masks(img_path, mask_path, src_path, threshold)
    #create_stems(img_path,mask_path,src_path,threshold)
    #ontop(img_path, mask_path, src_path, threshold)
    #filtter_images(img_path, mask_path, src_path, threshold, lower, upper)
    #
    #get_pred_histograms(img_path, mask_path, src_path, threshold,hist_type)







if __name__ == '__main__':
    main()
