import logging
import cv2
import numpy as np
# from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb  # , slic, quickshift, watershed
from .utils import Utils
from work.segmentation.clarifruit_segmentation.image import Image
from work.preprocess.display_functions import *
COLOR_DICT = {'gray':cv2.IMREAD_GRAYSCALE,'color':cv2.IMREAD_UNCHANGED}

logger = logging.getLogger(__name__)


class Segmentation:

    def __init__(self, img_path,mask_path=None,
                 scale=100, sigma=0.5, min_size=50, pr_threshold=0.05):

        logger.debug(" -> __init__")
        if type(img_path) == str:
            self.img= cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        else:
            self.img = img_path
        if mask_path is not None:
            if type(mask_path) == str:
                self.mask= cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)
            else:
                self.mask = mask_path

        else:
            self.mask = None
        self.segments = None
        self.boundaries = None
        self.segments_count = 0
        self.segments_hsv = None

        self.filtered_segments = None


        self.pr_threshold = pr_threshold
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size


        logger.debug(" <- __init__")

    def get_segments(self):
        float_image = img_as_float(self.img)
        return felzenszwalb(float_image, scale=self.scale, sigma=self.sigma, min_size=self.min_size)


    def apply_segmentation(self):

        self.segments = self.get_segments()
        segments = np.unique(self.segments)
        self.segments_count = len(segments)

        if self.mask is not None:
            self.filter_segments()




    def segment_iterator(self):
        n_segments = self.segments.max()
        for i in range(n_segments):
            segment = np.where(self.segments == i, True, False)
            #segment = segment[...,np.newaxis]
            yield i, segment

    def filter_segments(self):
        self.filtered_segments = np.zeros_like(self.segments,dtype=np.bool)
        for i, segment in self.segment_iterator():
            seg_sum = 255 * np.count_nonzero(segment)
            segment_activation = self.mask[segment]
            #segment_activation = np.bitwise_and(self.image.mask_resized_binary, segment)
            seg_activation_sum = np.sum(segment_activation)
            activation_pr = (seg_activation_sum / seg_sum)
            if activation_pr > self.pr_threshold:
                self.filtered_segments[segment] = True


    def return_modified_mask(self):
        return self.binary_to_grayscale(self.filtered_segments)




    @staticmethod
    def binary_to_grayscale(img):
        res = img.copy()
        res = (255 * res).astype(np.uint8)
        return res






