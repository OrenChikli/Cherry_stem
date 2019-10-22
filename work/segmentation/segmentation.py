import logging
from skimage.util import img_as_float

from skimage.segmentation import felzenszwalb, slic, quickshift
import cv2

from work.auxiliary.custom_image import CustomImage

from work.auxiliary.decorators import Logger_decorator
from work.auxiliary import data_functions
import os
from datetime import datetime
import numpy as np

COLOR_DICT = {'gray': cv2.IMREAD_GRAYSCALE, 'color': cv2.IMREAD_UNCHANGED}
SEG_ALGORITHMS_DICT = {'felzenszwalb': felzenszwalb,
                       'slic': slic,
                       'quickshift': quickshift}


logger = logging.getLogger(__name__)
logger_decorator = Logger_decorator(logger=logger)


class SegmentationSingle(CustomImage):
    """
    A class for performing segmentation enhancement on given segmentation masks
    """

    @logger_decorator.debug_dec
    def __init__(self, seg_type='felzenszwalb', seg_params=None,
                 pr_threshold=0.05, gray_scale=False, **kwargs):

        super().__init__(**kwargs)
        self.segments = None
        self.segmentation_mask = None

        self.pr_threshold = pr_threshold
        self.gray_scale = gray_scale
        self.seg_type = seg_type
        self.seg_params = seg_params

    @logger_decorator.debug_dec
    def get_segments(self):
        """
        Apply one of the segmentation algorithms specified in
        SEG_ALOGORITHEMS_DICT (all algorithems are from the skimage.segmentation
         module)according to the "seg_type"
         if "self.gray_scale" is True, will segment a grayscale image
        :return: the result of the segmentation algorithm
        """

        logger.info(
            f"performing {self.seg_type} image segmentation on {self.img_name}")
        if self.gray_scale:
            float_image = img_as_float(
                cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY))
        else:
            float_image = img_as_float(self.img)

        res = SEG_ALGORITHMS_DICT[self.seg_type](float_image,
                                                 **self.seg_params)
        return res

    # TODO: try and improve running time
    def segmentor(self):
        """
        decompose a given segmentation image of (width_height) where each pixel
         has a label, and a total of #n labels (e.g 1000), to a matrix of size
         (#n, width, height) where each row i corresponds to the representation
         of the i'th segment
        :return: a segmentation "matrix"
        """

        n_segments = self.segments.max()
        logger.info(f"found {n_segments} segments")
        seg_array = np.arange(n_segments).reshape(1, -1)
        res = np.equal(self.segments.T[...,np.newaxis],seg_array).T

        # dims = (n_segments, *self.img.shape[:-1])
        # res = np.zeros(dims, np.bool)
        # for i in range(n_segments):
        #     segment = np.where(self.segments == i, True, False)
        #     res[i] = segment
        return res

    def fillter_segments(self, save_flag=False):
        """
        Use a given image segmentation mask and augment it by using the results
        of an image segmentation algorithm.
        Overlays the segments of the algorithm that have a high degree of
        ovelap (specified by activation_pr) above a given threshold
        (self.pr_threshold)
        :param save_flag: bool, whether to save the activated segments,
        default False
        :return: augmented binary mask image
        """
        filtered_segments = self.segmentor()

        seg_sum = np.sum(filtered_segments, axis=(1, 2))

        #bin_mask = self.binary_mask > 150  # some masks where created strangely,
        # not only 255 or 0

        segment_activation = filtered_segments * self.binary_mask.astype(np.bool)
        seg_activation_sum = np.sum(segment_activation, axis=(1, 2))

        activation_pr = (seg_activation_sum / seg_sum)
        #self.pr_threshold= np.quantile(activation_pr, 0.97)
        res = filtered_segments[np.where(activation_pr > self.pr_threshold)]


        if save_flag:
            save_path = data_functions.create_path(self.save_path,
                                                   'active_segments')
            for i, seg in enumerate(res):
                save_name = f"seg_{i}.jpg"
                curr_save_path = os.path.join(save_path, save_name)
                img = (255 * seg).astype(np.uint8)
                cv2.imwrite(curr_save_path, img)

        res = res.sum(axis=0)
        res += 1 * self.binary_mask
        res = res.astype(np.bool)
        filtered_segments = (255 * res).astype(np.uint8)
        # filtered_segments = self.filtter_size(filtered_segments)

        return filtered_segments

    # TODO: think if this is needed
    @staticmethod
    def filter_size(img, min_size=100):
        """
        ----EXPERIMENTAL----

        :param img:
        :param min_size:
        :return:
        """
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            img, connectivity=8)

        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        img2 = np.zeros((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        return img2

    @logger_decorator.debug_dec
    def apply_segmentation(self, save_flag='stamped', display_flag=False,
                           save_segments=False):
        """
        Apply the segmentation augmentation on the given image and it's mask
        :param save_flag: bool, whether to save the augmented image result
        :param display_flag: bool, whether to dispaly the augmented image result
        :param save_segments: bool, whether to save activated segments
        :return:
        """
        logger.info(f"getting segmentation mask for img {self.img_name}")
        self.segments = self.get_segments()
        logger.info("performing mask improvment via segments")
        self.segmentation_mask = self.fillter_segments(
            save_flag=save_segments)

        if save_flag is not None:
            label = ''
            if save_flag == 'stamped':
                label = 'seg_mask'
                if self.create_save_dest_flag:
                    save_dict = {'pr_threshold': self.pr_threshold,
                                 'seg_type': self.seg_type,
                                 'seg_params': self.seg_params,
                                 "graysclae": self.gray_scale}
                    data_functions.save_json(save_dict,
                                             "segmentation_settings.json",
                                             self.save_path)

            self.save_img(self.segmentation_mask, label)

        if display_flag:
            label = 'Segmentation mask'
            self.display_img(self.segmentation_mask, label)

    @logger_decorator.debug_dec
    def get_ontop_seg(self, mask_color=(255, 0, 0), display_flag=False,
                  save_flag=False):
        """
        A modiffication of the CustomImageMethod for displaying the augmented
        segmentation mask ontop of the input image
        """
        ontop = super().get_ontop(mask_color=mask_color,
                                  mask=self.segmentation_mask,
                                  display_flag=display_flag,
                                  disp_label="Segmentation Mask ontop the image",
                                  save_flag=save_flag,
                                  save_label='seg_ontop')

        return ontop


@logger_decorator.debug_dec
def segment_multi(img_path, mask_path, save_path, is_binary_mask=True,
                  settings_dict=None, img_list=None,create_stamp=True):
    """
    Preform segmentation augmentation on multiple images using the
    :param img_path: str, the source path to the images
    :param mask_path:  str, the source path to the maskes
    :param save_path:  str , the path to save the results
    :param is_binary_mask: bool, specify if the masks are binary or raw npy
     files given as a results of the unet model
    :param settings_dict: settings for the SegmentationSingle e.g
    {'pr_threshold': 0.15,
                     'seg_type':"felzenszwalb",
                     'seg_params': {'scale': 1, 'sigma': 0,'min_size': 5},
                     'gray_scale': False}
    :param img_list: optional, a list of specific names in the img_path to be
    augmented
    :return:
    """

    img_list = os.listdir(img_path) if img_list is None else img_list

    if create_stamp=='stamped':
        dir_save_path = data_functions.create_path(save_path, 'several')
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_save_path = data_functions.create_path(dir_save_path, current_time)
        data_functions.save_json(settings_dict, "segmentation_settings.json",
                                 dir_save_path)

    else:
        dir_save_path = save_path


    logger.info(f"segmenting to {dir_save_path}")
    save_path = dir_save_path
    for img_name in img_list:

        curr_img_path = os.path.join(img_path, img_name)
        if is_binary_mask:
            curr_mask_path = os.path.join(mask_path, img_name)

        else:
            mask_name = img_name.rsplit('.', 1)[0] + '.npy'
            curr_mask_path = os.path.join(mask_path, mask_name)

        sg = SegmentationSingle(img_path=curr_img_path,
                                mask_path=curr_mask_path,
                                is_binary_mask=is_binary_mask,
                                save_path=save_path,
                                create_save_dest_flag=False,
                                **settings_dict)
        sg.apply_segmentation(save_flag='raw')
        sg.get_ontop_seg(save_flag=True)

