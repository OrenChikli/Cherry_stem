import logging
from skimage.util import img_as_float

from skimage.segmentation import felzenszwalb

from work.auxiliary.image import Image
from work.auxiliary.display_functions import *

COLOR_DICT = {'gray': cv2.IMREAD_GRAYSCALE,'color': cv2.IMREAD_UNCHANGED}


logger = logging.getLogger(__name__)


class Segmentation:

    def __init__(self, img_path,mask_path=None,
                 scale=100, sigma=0.5, min_size=50, pr_threshold=0.05):


        logger.debug(" ->__init__")

        self.img = Image(img_path, mask_path)

        self.boundaries = None
        self.filtered_segments = None

        self.pr_threshold = pr_threshold
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size

        self.apply_segmentation()

        logger.debug(" <- __init__")

    def get_segments(self):
        float_image = img_as_float(self.img.img)
        return felzenszwalb(float_image, scale=self.scale, sigma=self.sigma, min_size=self.min_size)

    def apply_segmentation(self):
        logger.debug(" -> apply_segmentation")
        self.segments = self.get_segments()
        if self.img.grayscale_mask is not None:
            self.filter_segments()
        logger.debug(" <- apply_segmentation")


    def segment_iterator(self):
        n_segments = self.segments.max()
        for i in range(n_segments):
            segment = np.where(self.segments == i, True, False)
            yield i, segment

    def filter_segments(self):


        logger.debug(" -> filter_segments")
        self.filtered_segments = np.zeros_like(self.segments,dtype=np.bool)
        for i, segment in self.segment_iterator():
            seg_sum = 255 * np.count_nonzero(segment)
            segment_activation = self.img.grayscale_mask[segment]
            seg_activation_sum = np.sum(segment_activation)
            activation_pr = (seg_activation_sum / seg_sum)
            if activation_pr > self.pr_threshold:
                self.filtered_segments[segment] = True
        logger.debug(" <- filter_segments")


    def return_modified_mask(self):
        return self.binary_to_grayscale(self.filtered_segments)


    @staticmethod
    def binary_to_grayscale(img):
        res = img.copy()
        res = (255 * res).astype(np.uint8)
        return res






