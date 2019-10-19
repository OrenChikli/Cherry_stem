import logging
from skimage.util import img_as_float

from skimage.segmentation import felzenszwalb,slic, quickshift
import cv2

from work.auxiliary.custom_image import CustomImage

from work.auxiliary.decorators import Logger_decorator
from work.auxiliary import data_functions
import os
from datetime import datetime
import numpy as np

COLOR_DICT = {'gray': cv2.IMREAD_GRAYSCALE, 'color': cv2.IMREAD_UNCHANGED}
SEG_ALOGORITHEMS_DICT = {'felzenszwalb':felzenszwalb,
                         'slic':slic,
                         'quickshift': quickshift}


logger = logging.getLogger(__name__)
logger_decorator = Logger_decorator(logger=logger,log_type="DEBUG")


class SegmentationSingle(CustomImage):

    @logger_decorator.debug_dec
    def __init__(self,seg_type='felzenszwalb',seg_params=None,
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

        logger.info(f"performing {self.seg_type} image segmentation on {self.img_name}")
        if self.gray_scale:
            float_image = img_as_float(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY))
        else:
            float_image = img_as_float(self.img)

        res = SEG_ALOGORITHEMS_DICT[self.seg_type](float_image, **self.seg_params)
        return res

    def segmentor(self):
        n_segments = self.segments.max()
        dims = (n_segments,*self.img.shape[:-1])
        res = np.zeros(dims,np.bool)
        for i in range(n_segments):
            segment = np.where(self.segments == i, True, False)
            res[i] = segment
        return res

    def fillter_segments_improved(self,save_flag=False):
        filtered_segments = self.segmentor()

        seg_sum = np.sum(filtered_segments,axis=(1,2))

        bin_mask =self.binary_mask > 150 # some masks where created strangely, not only 255 or 0

        segment_activation = filtered_segments * bin_mask
        seg_activation_sum = np.sum(segment_activation,axis=(1,2))

        activation_pr = (seg_activation_sum / seg_sum)

        res = filtered_segments[np.where(activation_pr > self.pr_threshold)]

        if save_flag:
            save_path = data_functions.create_path(self.save_path,'active_segments')
            for i,seg in enumerate(res):
                save_name = f"seg_{i}.jpg"
                curr_save_path = os.path.join(save_path,save_name)
                img = (255 * seg).astype(np.uint8)
                cv2.imwrite(curr_save_path,img)

        res = res.sum(axis=0)
        res += 1*bin_mask
        res = res.astype(np.bool)
        filtered_segments = (255 * res).astype(np.uint8)
        #filtered_segments = self.filtter_size(filtered_segments)

        return filtered_segments
    @staticmethod
    def filtter_size(img,min_size=100):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        img2 = np.zeros((output.shape))
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        return img2

    @logger_decorator.debug_dec
    def segment_iterator(self):
        n_segments = self.segments.max()
        for i in range(n_segments):
            segment = np.where(self.segments == i, True, False)
            yield i, segment



    @logger_decorator.debug_dec
    def apply_segmentation(self,save_flag=False,display_flag=False,save_segments=False):
        logger.info(f"getting segmentation mask for img {self.img_name}")
        self.segments = self.get_segments()
        logger.info("performing mask improvment via segments")
        self.segmentation_mask = self.fillter_segments_improved(save_flag=save_segments)


        if save_flag:
            label = 'seg_mask'
            self.save_img(self.segmentation_mask, label)
            save_dict = {'pr_threshold': self.pr_threshold,
                         'seg_type':self.seg_type,
                         'seg_params':self.seg_params,
                         "graysclae": self.gray_scale}
            data_functions.save_json(save_dict, "segmentation_settings.json", self.save_path)

        if display_flag:
            label = 'Segmentation mask'
            self.display_img(self.segmentation_mask,label)



    @logger_decorator.debug_dec
    def get_ontop(self,mask_color=(255,0,0),display_flag=False,save_flag=False):

        ontop = super().get_ontop(mask_color=mask_color,
                                  mask=self.segmentation_mask,
                                  display_flag=display_flag,
                                  disp_label="Segmentation Mask ontop the image",
                                  save_flag=save_flag,
                                  save_label='seg_ontop')

        return ontop




class SegmentationMulti:
    @logger_decorator.debug_dec
    def __init__(self, img_path, mask_path, seg_path, is_binary_mask=True):
        self.img_path = img_path
        self.mask_path = mask_path
        self.is_binary_mask = is_binary_mask
        self.save_path = seg_path

    @logger_decorator.debug_dec
    def segment_multi(self, settings_dict, img_list=None):

        if img_list is None:
            img_list = [img_entry.name for img_entry in os.scandir(self.img_path)]

        dir_save_path = data_functions.create_path(self.save_path, 'several')
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_save_path = data_functions.create_path(dir_save_path, current_time)

        data_functions.save_json(settings_dict, "segmentation_settings.json", dir_save_path)

        logger.info(f"segmenting to {dir_save_path}")
        self.save_path = dir_save_path
        for img_name in img_list:

            curr_img_path = os.path.join(self.img_path, img_name)
            if self.is_binary_mask:
                curr_mask_path = os.path.join(self.mask_path, img_name)

            else:
                mask_name = img_name.rsplit('.',1)[0]+'.npy'
                curr_mask_path = os.path.join(self.mask_path, mask_name)

            sg = SegmentationSingle(img_path=curr_img_path, mask_path=curr_mask_path,
                                    is_binary_mask=self.is_binary_mask,
                                    save_path=self.save_path,
                                    create_save_dest_flag=False, **settings_dict)
            sg.apply_segmentation(save_flag=True)
            sg.get_ontop(save_flag=True)



