from work.stem_extraction.stem_extract import *

from logger_settings import *

configure_logger()
logger = logging.getLogger("extraction_main")


def get_binary_masks(img_path, mask_path, save_path, threshold):
    """
    create binary masks via the given threshold of the makes given in the mask_path and save them in the src_path
    :param img_path: path to source images
    :param mask_path:path to the unet maskes of the source images
    :param save_path: the save path of the results
    :param threshold: the threshold to get the binary mask, the binary mask will be activated for pixels that have
    values greater than the threshold
    :return:
    """

    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    save_path=save_path,
                                    threshold=threshold)

    logger.info(f"getting binary masks with threshold: {threshold} from{mask_path}")
    logger.info(f"saving results at {save_path}")

    stem_exctractor.get_threshold_masks()


def create_stems(img_path, mask_path, save_path, threshold, use_thres_flag=True,hist_type='bgr'):
    """
    extract the stems from the source images using the unet maskes. first binary maskes are created via the threshold,
    than the maskes are used to select the stems from the source images in a new image
    all the stems are saved in the save_path
    :param img_path: path to source images
    :param mask_path:path to the unet masks of the source images
    :param save_path: the save path of the results
    :param threshold: the threshold to get the binary mask, the binary mask will be activated for pixels that have
    values greater than the threshold
    :return:
    """
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    save_path=save_path,
                                    threshold=threshold,
                                    use_thres_flag=use_thres_flag)

    logger.info(f"getting stems for threshold {threshold}")
    logger.info(f"saving results at {save_path}")

    stem_exctractor.get_stems()


def ontop(img_path, mask_path, save_path, threshold,use_thres_flag=True,hist_type='bgr'):
    """
    draw the binary maskes ontop of the source images for visual inspection
    :param img_path: path to source images
    :param mask_path:path to the unet masks of the source images
    :param save_path: the save path of the results
    :param threshold: the threshold to get the binary mask, the binary mask will be activated for pixels that have
    values greater than the threshold
    :return:
    """
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    save_path=save_path,
                                    threshold=threshold,
                                    use_thres_flag=use_thres_flag)

    logger.info(f"getting ontop images for threshold {threshold}")
    logger.info(f"saving results at {save_path}")

    stem_exctractor.get_ontop_images()


def filter_images(img_path, mask_path, save_path, threshold, hist_type='bgr', use_thres_flag=True):
    """
    a method to try and color filter the extracted stems using a lowwer and upper color
    threshold, will result in an image where only color in between lower and upper are shown
    :param img_path: path to source images
    :param mask_path:path to the unet maskes of the source images
    :param save_path: the save path of the results
    :param threshold: the threshold to get the binary mask, the binary mask will be activated for pixels that have
    values greater than the threshold
    :param lower: the lower color boundary to create the color mask
    :param upper the upper color boundary to create the upper color mask
    :return:
    """
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    save_path=save_path,
                                    threshold=threshold,
                                    use_thres_flag=use_thres_flag)

    logger.info(f"getting filterred images for threshold {threshold}")
    logger.info(f"saving results at {save_path}")

    stem_exctractor.fillter_via_color()



def get_pred_histograms(img_path, mask_path, save_path, threshold, hist_type='bgr', use_thres_flag=True):
    """
    get color histograms of the source images using the unet maskes, i.e get color histograms of the
    areas activated in the binary mask
    :param img_path: path to source images
    :param mask_path:path to the unet masks of the source images
    :param save_path: the save path of the results
    :param threshold: the threshold to get the binary mask, the binary mask will be activated for pixels that have
    values greater than the threshold
    :param hist_type: the type of histogram to calculate. default value is for 'bgr' which is the normal color mode.
    another option is 'hsv' for heu,saturation and value color space
    :return:
    """
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    save_path=save_path,
                                    threshold=threshold,
                                    use_thres_flag=use_thres_flag)

    logger.info(f"getting {hist_type} histograms for threshold {threshold}")
    logger.info(f"saving results at {save_path}")

    stem_exctractor.calc_hists(hist_type=hist_type)


FUNC_DICTIPNARY = {'binary': get_binary_masks,
                   'hists': get_pred_histograms,
                   'ontop': ontop,
                   'stems': create_stems,
                   'filter_images':filter_images}


def create_raw_ground_truth(img_path, mask_path, dest_path, train_folder='classifier_train_data'):
    """
    getting respective masks of images in diffrent classes.
    a method to get the predictions from src_mask_path, into the y_folder in the ground_path, where the x_folder has
    the source images, selected by the user
    :param img_path: a path containig the ground truth, has structure of x_folder with src images, y_folder with labels
    :param mask_path: the image where all the avaliabel predictions reside
    :param x_folder: the name of the subfolder in ground truth which has the src images
    :param y_folder: the name of the subfolder in the ground truth path which will house the labels
    :return: None
    """
    curr_dest_path = data_functions.create_path(dest_path, train_folder)
    for curr_class in os.scandir(img_path):
        curr_dest = data_functions.create_path(curr_dest_path, curr_class.name)

        logger.info(f"getting ground truth for class {curr_class.name}")
        logger.info(f"copying ground truth from {curr_class.path}")
        logger.info(f"copying ground truth to {curr_dest}")

        data_functions.get_masks_via_img(curr_class.path, mask_path, curr_dest)


def create_ground_truth_objects(ground_path, threshold, src_path, obj_type,
                                train_folder='classifier_train_data', hist_type='bgr'):

    dest_path = data_functions.create_path(src_path, f"thres_{threshold}")
    dest_path = data_functions.create_path(dest_path, train_folder)
    raw_pred_path = os.path.join(src_path, train_folder)

    func = FUNC_DICTIPNARY[obj_type]

    for curr_class in os.scandir(raw_pred_path):
        curr_raw_pred_path = os.path.join(raw_pred_path, curr_class.name)
        curr_dest = data_functions.create_path(dest_path, curr_class.name)
        curr_ground_path = os.path.join(ground_path, curr_class.name)

        func(img_path=curr_ground_path,
             mask_path=curr_raw_pred_path,
             save_path=curr_dest,
             threshold=threshold,
             hist_type=hist_type,
             use_thres_flag=False)




def get_test_train():
    src_path = r'D:\Clarifruit\cherry_stem\data\classification_data\from_all\set3\All'
    dest_path = r'D:\Clarifruit\cherry_stem\data\classification_data\from_all\set3'
    data_functions.get_train_test_split(src_path, dest_path, train_name='train', test_name='test', test_size=0.33)


def main():
    src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-10-11_18-09-39'

    train_folder = 'train'
    test_folder = 'test'
    raw_preds_folder = 'raw_pred'
    mask_path = os.path.join(src_path, raw_preds_folder)
    img_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    ground_path = r'D:\Clarifruit\cherry_stem\data\classification_data\from_all\set1'
    ground_train_path = os.path.join(ground_path, train_folder)
    ground_test_path = os.path.join(ground_path, test_folder)

    threshold = 0.4
    hist_type = 'hsv'
    object_type='ontop'

    save_flag = True

    # create train data
    # create_raw_ground_truth(ground_train_path, mask_path, src_path,
    #                     train_folder=train_folder)

    # create_ground_truth_objects(ground_path=ground_train_path,
    #                             threshold=threshold,
    #                             src_path=src_path,
    #                             train_folder=train_folder,
    #                             hist_type=hist_type,
    #                             obj_type=object_type)

    # #create test data
    # create_raw_ground_truth(ground_test_path, mask_path, src_path,
    #                     train_folder=test_folder)

    # create_ground_truth_objects(ground_path=ground_test_path,
    #                             threshold=threshold,
    #                             src_path=src_path,
    #                             train_folder=test_folder,
    #                             hist_type=hist_type,
    #                             obj_type=object_type)
    # experiment with current data

    get_binary_masks(img_path, mask_path, src_path, threshold)
    # create_stems(img_path,mask_path,src_path,threshold)
    ontop(img_path, mask_path, src_path, threshold)
    # filtter_images(img_path, mask_path, src_path, threshold, save_flag)
    #get_pred_histograms(img_path, mask_path, src_path, threshold, hist_type)


if __name__ == '__main__':
    # get_test_train()
    main()
