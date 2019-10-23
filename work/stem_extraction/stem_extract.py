from work.auxiliary import data_functions
import os
import cv2
import numpy as np
import shutil
import logging
from work.auxiliary import decorators
from work.segmentation.segmentation import SegmentationSingle

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logger_decorator = decorators.Logger_decorator(logger)


OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Chi-Squared", cv2.HISTCMP_CHISQR),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

HIST_TYPE = {'rgb': lambda x: x.get_hist_via_mask,
             'hsv': lambda x: x.get_hsv_hist}

FUNC_DICTIPNARY = {'binary': lambda x: x.get_threshold_masks(),
                   'hists': lambda x: x.calc_hists(),
                   'ontop': lambda x: x.get_ontop_images(),
                   'stems': lambda x: x.get_stems(),
                   'filter_images': lambda x: x.filter_images()}


class StemExtractor:
    """
        get color histograms of the source images using the unet maskes,
        i.e get color histograms of the areas activated in the binary mask

    """

    @logger_decorator.debug_dec
    def __init__(self, img_path, mask_path, save_path, threshold=0.5,
                 use_thres_flag=True, hist_type='bgr', is_binary_mask=False):
        """

        :param img_path: path to source images
        :param mask_path:path to the unet masks of the source images
        :param save_path: the save path of the results
        :param threshold: the threshold to get the binary mask, the binary mask will be activated for pixels that have
        values greater than the threshold

        :param use_thres_flag:
        :param hist_type: the type of histogram to calculate. default value is for 'bgr' which is the normal color mode.
        another option is 'hsv' for heu,saturation and value color space
        """

        self.img_path = img_path
        self.mask_path = mask_path
        self.threshold = threshold
        self.is_binary_mask = is_binary_mask
        if use_thres_flag:
            self.thres_save_path = data_functions.create_path(save_path,
                                                              f"thres_{threshold}")
        else:
            self.thres_save_path = save_path

        self.threshold_masks_path = None

        self.cut_image_path = None
        self.ontop_path = None
        self.hist_type = hist_type

    @logger_decorator.debug_dec
    def get_threshold_masks(self):
        """
        get the binary mask, where the pixel value of the source mask (float
        type) are greater than self.threshold
        :return:
        """
        logger.info(
            f"getting binary masks with threshold: {self.threshold} "
            f"from{self.mask_path}")

        self.threshold_masks_path = data_functions.create_path(
            self.thres_save_path, f'binary')

        logger.info(f"saving results at {self.threshold_masks_path}")
        for img in self.image_obj_iterator():
            img_save_path = os.path.join(self.threshold_masks_path,
                                         img.image_name)
            cv2.imwrite(img_save_path, img.threshold_mask)

    @logger_decorator.debug_dec
    def get_stems(self):
        """
        extract the "stems" from the image - return the areas in the image that are activated in the thresholded mask
        eg if pixel (156,46) is turned on in the mask image, it will show in the result
        :return:
        """

        self.cut_image_path = data_functions.create_path(self.thres_save_path,
                                                         f'stems')
        logger.info(f"getting stems for threshold {self.threshold}")
        logger.info(f"creting stems in {self.cut_image_path}")

        for img in self.image_obj_iterator():
            image_cut = img.cut_via_mask()
            cv2.imwrite(os.path.join(self.cut_image_path, img.image_name),
                        image_cut)

    @logger_decorator.debug_dec
    def get_ontop_images(self):
        """
        return the overlay of the thresholded mask on top of the source image
        :return:
        """

        self.ontop_path = data_functions.create_path(self.thres_save_path,
                                                     f'on_top')
        logger.info(f"getting ontop images for threshold {self.threshold}")
        logger.info(f"creting ontop images in {self.ontop_path}")
        for img in self.image_obj_iterator():
            cv2.imwrite(os.path.join(self.ontop_path, img.img_name),
                        img.get_ontop())

    @logger_decorator.debug_dec
    @staticmethod
    def get_label(pr_green, pr_brown, img_name):
        """
        ---EXPERIMENTAL---
        :param pr_green:
        :param pr_brown:
        :param img_name:
        :return:
        """
        label = 'D'
        if pr_green >= 0.5:
            if pr_brown < 0.1:
                label = 'A'
            if pr_brown > 0.1:
                label = 'B'
        elif 0.5 > pr_green > 0.1:
            p = pr_brown / pr_green
            if p < 0.25:
                label = 'A'
            elif 0.25 <= p < 0.6:
                label = 'B'
            else:
                label = 'C'
        x = label
        return label

    """    def get_label(pr_green,pr_brown):
        label = 'D'
        if pr_green >= 0.5 and pr_brown < 0.2: label ='A'
        elif  0.5 > pr_green >= 0.3 and 0.3 >pr_brown >= 0.2: label = 'B'
        elif   0.3 > pr_green and 0.4 > pr_brown >= 0.3: label = 'C'
        return label"""

    @logger_decorator.debug_dec
    @staticmethod
    def get_label1(pr):
        """
        ---EXPERIMENTAL---
        :param pr:
        :return:
        """
        label = 'A'
        if pr >= 0.5:
            label = 'D'
        elif 0.5 > pr >= 0.3:
            label = 'C'
        elif 0.3 > pr >= 0.1:
            label = 'B'
        return label

    # def fillter_via_color(self,save_flag=False):
    #     logger.debug(" <- fillter_via_color")
    #     out_path = data_functions.create_path(self.thres_save_path, f'filtered')
    #     logger.info(f"creting filltered images in {out_path}")
    #     for img in tqdm(self.image_obj_iterator()):
    #         pr_green,pr_brown = img.filter_cut_iamge()
    #         raw_name = img.image_name.rsplit('.',1)[0]
    #         pred = self.get_label(pr_green,pr_brown,img.image_name)
    #         curr_save_path = data_functions.create_path(out_path, pred)
    #         _ = shutil.copy(img.img_path, curr_save_path)
    #         if save_flag:
    #             cv2.imwrite(os.path.join(out_path, f'{raw_name}_green.jpg'), img.green_part)
    #             cv2.imwrite(os.path.join(out_path, f'{raw_name}_brown.jpg'), img.brown_part)
    #             cv2.imwrite(os.path.join(out_path, img.image_name), img.threshold_mask)
    #     logger.debug(" -> get_notop")

    @logger_decorator.debug_dec
    def fillter_via_color_green_brown(self, save_flag=False):
        """
        ---EXPERIMENTAL---
        :param save_flag:
        :return:
        """
        out_path = data_functions.create_path(self.thres_save_path, f'filtered')
        logger.info(f"creting filltered images in {out_path}")
        for img in self.image_obj_iterator():
            pr_green, pr_brown = img.filter_cut_image_green_brown()
            raw_name = img.image_name.rsplit('.', 1)[0]
            pred = self.get_label(pr_green, pr_brown, img.image_name)
            curr_save_path = data_functions.create_path(out_path, pred)
            _ = shutil.copy(img.img_path, curr_save_path)
            if save_flag:
                cv2.imwrite(os.path.join(out_path, f'{raw_name}_green.jpg'),
                            img.green_part)
                cv2.imwrite(os.path.join(out_path, f'{raw_name}_brown.jpg'),
                            img.brown_part)
                cv2.imwrite(os.path.join(out_path, img.image_name),
                            img.threshold_mask)

    @logger_decorator.debug_dec
    def fillter_via_color(self):
        """
        ---EXPERIMENTAL---
        :return:
        """

        out_path = data_functions.create_path(self.thres_save_path, f'filtered')
        logger.info(f"getting filterred images for threshold {self.threshold}")
        logger.info(f"creting filltered images in {out_path}")
        for img in self.image_obj_iterator():
            res = img.filter_cut_image()
            cv2.imwrite(os.path.join(out_path, img.image_name), res)

    @logger_decorator.debug_dec
    def image_obj_iterator(self):
        """
        an iterator of image objects for performing variuos actions,
        yields an instance of CustomImageExtractor
        :return:
        """
        for img_entry in os.scandir(self.img_path):
            if not self.is_binary_mask:  # the source are results of unet- has
                # float values form 0 to 1
                mask_name = img_entry.name.rsplit(".", 1)[0] + '.npy'
            else:  # the mask are already binary (0 or 255)
                mask_name = img_entry.name
            mask_path = os.path.join(self.mask_path, mask_name)
            img = CustomImageExtractor(img_path=img_entry.path,
                                       mask_path=mask_path,
                                       is_binary_mask=self.is_binary_mask,
                                       threshold=self.threshold,
                                       hist_type=self.hist_type,
                                       create_save_dest_flag=False)

            yield img

    @logger_decorator.debug_dec
    def calc_hists(self):
        """
        A method to calculate color histograms on source images while using
        the segmentation mask for calculation at the segmentation areas
        :return:
        """
        dest_path = data_functions.create_path(self.thres_save_path,
                                               f'{self.hist_type}_histograms')
        logger.info(
            f"getting {self.hist_type} histograms for threshold"
            f" {self.threshold}")
        logger.info(f"saving results at {dest_path}")
        for img in self.image_obj_iterator():
            curr_dest_path = os.path.join(dest_path, f"{img.img_raw_name}.npy")
            fig_big_hist = img.get_hist_via_mask()
            np.save(curr_dest_path, fig_big_hist)


@logger_decorator.debug_dec
def create_object(img_path, mask_path, save_path, threshold, hist_type,
                  use_thres_flag,
                  obj_type, is_binary_mask):
    stem_exctractor = StemExtractor(img_path=img_path,
                                    mask_path=mask_path,
                                    save_path=save_path,
                                    threshold=threshold,
                                    use_thres_flag=use_thres_flag,
                                    hist_type=hist_type,
                                    is_binary_mask=is_binary_mask)

    FUNC_DICTIPNARY[obj_type](stem_exctractor)


@logger_decorator.debug_dec
def create_ground_truth_objects(ground_path,mask_path,save_path, threshold,
                                hist_type, obj_type,use_thres_flag,
                                is_binary_mask,folder='train'):
    """
    A method to create object from the ground truth path,
    e.g can create hsv histogrames for the test and train,
    or bgr histogrames, or return images of the masks ovelayed ontop of the
    source images . e.t.c
    :param ground_path: path the the groud truth test and train dataset
    :param mask_path: path to the masks which will be used
    :param save_path:  the destination path to save the results
    :param threshold: float, if the mask are the results of a prediction,
    than this is used to create binary images
    :param hist_type: str,optional, if creating a histogram, what type of
     histogram, 'bgr' or 'hsv'
    :param obj_type: str, what type of object to create , binary_images,histograms,
    stems (cutting the images with the masks)
    :param use_thres_flag: bool, where to create a new save folder in the sestination
    path for the current instance
    :param is_binary_mask: bool, whether the masks are binary
    :param folder: str, where this instance is for test or train
    :return:
    """
    if use_thres_flag :
        dest_path = data_functions.create_path(save_path, f"thres_{threshold}")
    dest_path = data_functions.create_path(dest_path, folder)
    ground_path = os.path.join(ground_path, folder)
    logger.info(f"getting {obj_type} objects for {folder}")
    for curr_class in os.scandir(ground_path):
        logger.info(f"getting objects for {curr_class.name} class")
        curr_dest = data_functions.create_path(dest_path, curr_class.name)
        curr_ground_path = os.path.join(ground_path, curr_class.name)

        create_object(img_path=curr_ground_path,
                      mask_path=mask_path,
                      save_path=curr_dest,
                      threshold=threshold,
                      hist_type=hist_type,
                      use_thres_flag=False,
                      obj_type=obj_type,
                      is_binary_mask=is_binary_mask)


def create_test_train_obj(ground_path, mask_path, save_path, threshold,
                          hist_type,
                          use_thres_flag, obj_type, is_binary_mask):
    """

    :param ground_path:
    :param mask_path:
    :param save_path:
    :param threshold:
    :param hist_type:
    :param use_thres_flag:
    :param obj_type:
    :param is_binary_mask:
    :return:
    """

    create_ground_truth_objects(ground_path=ground_path,
                                mask_path=mask_path,
                                save_path=save_path,
                                threshold=threshold,
                                hist_type=hist_type,
                                use_thres_flag=use_thres_flag,
                                obj_type=obj_type,
                                is_binary_mask=is_binary_mask,
                                folder='train')

    create_ground_truth_objects(ground_path=ground_path,
                                mask_path=mask_path,
                                save_path=save_path,
                                threshold=threshold,
                                hist_type=hist_type,
                                obj_type=obj_type,
                                use_thres_flag=use_thres_flag,
                                is_binary_mask=is_binary_mask,
                                folder='test')


class CustomImageExtractor(SegmentationSingle):
    def __init__(self, hist_type='hsv', **kwargs):
        super().__init__(**kwargs)
        self.hist_type = hist_type

    @logger_decorator.debug_dec
    def filter_cut_image(self):
        """
        ---Experimental---
        :return:
        """
        if self.image_cut is None:
            self.image_cut = self.cut_via_mask()

        hsv = cv2.cvtColor(self.image_cut, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (26, 40, 40), (86, 255, 255))
        mask_brown = cv2.inRange(hsv, (12, 50, 50), (20, 255, 255))
        mask = mask_brown + mask_green
        # mask = cv2.inRange(hsv, (12, 0, 0), (100, 255, 255))
        res = cv2.bitwise_and(self.image_cut, self.image_cut, mask=mask)
        return res

    @logger_decorator.debug_dec
    def filter_cut_image_green_brown(self):
        """
        ---Experimental---
        :return:
        """
        if self.image_cut is None:
            self.image_cut = self.cut_via_mask()

        hsv = cv2.cvtColor(self.image_cut, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (26, 40, 40), (86, 255, 255))
        mask_brown = cv2.inRange(hsv, (12, 50, 50), (20, 255, 255))
        mask_sum = np.sum(self.binary_mask)
        pr_green = np.sum(mask_green) / mask_sum
        pr_brown = np.sum(mask_brown) / mask_sum
        self.green_part = cv2.bitwise_and(self.image_cut, self.image_cut,
                                          mask=mask_green)
        self.brown_part = cv2.bitwise_and(self.image_cut, self.image_cut,
                                          mask=mask_brown)
        return pr_green, pr_brown

    @logger_decorator.debug_dec
    def get_hist_via_mask_cut(self, display_flag=False):
        """
                ---Experimental---

        :param hist_type:
        :param display_flag:
        :return:
        """
        img = self.img.copy()
        if self.hist_type == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv = cv2.cvtColor(self.cut_via_mask(), cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (26, 40, 40), (86, 255, 255))
        mask_brown = cv2.inRange(hsv, (12, 50, 50), (20, 255, 255))
        mask = mask_brown + mask_green

        hist = []
        for i in range(3):
            curr_hist = cv2.calcHist([img], [i], mask=mask, histSize=[256],
                                     ranges=[0, 256])
            hist.append(np.squeeze(curr_hist, axis=1))

        return np.array(hist)

    @logger_decorator.debug_dec
    def get_hist_via_mask(self, display_flag=False):
        """

        ---Experimental---

        :param hist_type:
        :param display_flag:
        :return:
        """
        img = self.img.copy()
        if self.hist_type == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = self.binary_mask.copy()
        masked_img = self.cut_via_mask()

        hist = []
        for i in range(3):
            curr_hist = cv2.calcHist([img], [i], mask=mask, histSize=[256],
                                     ranges=[0, 256])
            hist.append(np.squeeze(curr_hist, axis=1))

        if display_flag:
            fig, ax = plt.subplots(2, 2, figsize=(16, 10))
            ax_flat = ax.flatten()
            ax_flat[0].imshow(img[..., ::-1])
            ax_flat[1].imshow(np.squeeze(mask, axis=2), 'gray')
            ax_flat[2].imshow(masked_img[..., ::-1])
            if self.hist_type == 'bgr':
                ax_flat[3].plot(hist[0], label='blue', color='blue')
                plt.plot(hist[1], label='green', color='green')
                plt.plot(hist[2], label='red', color='red')
            else:
                ax_flat[3].plot(hist[0], label='hue', color='blue')
                plt.plot(hist[1], label='saturation', color='green')
                plt.plot(hist[2], label='value', color='red')
            plt.xlim([0, 256])
            plt.legend()
            plt.show()

        return np.array(hist)
