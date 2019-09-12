import logging
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb  # , slic, quickshift, watershed
#from Hawkeye.src.hawkeye.utils.file_utils import FileUtils
#from Hawkeye.src.hawkeye.cv.image import Image
from .image import Image

logger = logging.getLogger(__name__)


class MaskSegmentFinder:

    THRESHOLD_TRACKBAR_NAME = 'threshold'
    SCALE_TRACKBAR_NAME = 'Scale'
    SIGMA_TRACKBAR_NAME = 'Sigma'
    MSIZE_TRACKBAR_NAME = 'Min Size'

    IMAGE_WINDOW_NAME = 'SegmentFinder'

    def __init__(self, path,mask_path=None):
        logger.debug(" -> __init__")

        #self.mask_path=mask_path
        #if mask_path:
            #self.mask = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)

        self.path = path
        self.window_name = self.IMAGE_WINDOW_NAME + ' - ' + self.path
        #local_path = FileUtils.download_image_with_cache(path)
        self.image = Image(path,mask_path)
        self.image.prepare_for_detection()
        self.float_image = img_as_float(self.image.resized)

        self.segments = None
        self.boundaries = None
        self.segments_count = 0

        self.threshold = 1
        self.scale = 100
        self.sigma = 0.5
        self.min_size = 50


        # high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        logger.debug(" <- __init__")

    def on_trackbar_change(self, x):
        logger.debug('Trackbar - x: %d', x)
        self.apply()

    def apply(self):

        logger.debug(" -> apply")

        self.threshold = cv2.getTrackbarPos(MaskSegmentFinder.THRESHOLD_TRACKBAR_NAME, self.window_name)
        self.scale = cv2.getTrackbarPos(MaskSegmentFinder.SCALE_TRACKBAR_NAME, self.window_name)
        self.sigma = cv2.getTrackbarPos(MaskSegmentFinder.SIGMA_TRACKBAR_NAME, self.window_name) / 100
        self.min_size = cv2.getTrackbarPos(MaskSegmentFinder.MSIZE_TRACKBAR_NAME, self.window_name)

        self.segments = felzenszwalb(self.float_image, scale=self.scale, sigma=self.sigma, min_size=self.min_size)
        self.segments_count = len(np.unique(self.segments))
        self.filtered_segments =self.filter_segments(self.threshold)

        self.boundaries = mark_boundaries(self.image.resized, self.filtered_segments, color=(1, 1, 0))
        cv2.imshow(self.window_name, self.boundaries)

        logger.debug(" <- apply")

    def display(self):
        cv2.imshow(self.window_name, self.image.resized)
        cv2.createTrackbar(MaskSegmentFinder.THRESHOLD_TRACKBAR_NAME,self.window_name,1,100, self.on_trackbar_change)
        cv2.createTrackbar(MaskSegmentFinder.SCALE_TRACKBAR_NAME, self.window_name, 100, 1000, self.on_trackbar_change)
        cv2.createTrackbar(MaskSegmentFinder.SIGMA_TRACKBAR_NAME, self.window_name, 50, 100, self.on_trackbar_change)
        cv2.createTrackbar(MaskSegmentFinder.MSIZE_TRACKBAR_NAME, self.window_name, 50, 500, self.on_trackbar_change)

        self.apply()

    def segment_iterator(self):
        n_segments = self.segments.max()
        for i in range(n_segments):
            segment = np.where(self.segments == i, True, False)
            yield i, segment

    def filter_segments(self, threshold=1):
        res = np.zeros_like(self.segments, dtype=np.bool)
        for i, segment in self.segment_iterator():
            segment_activation = self.image.mask_resized * segment
            seg_sum = np.count_nonzero(segment_activation)
            if seg_sum >= threshold:
                res[segment] = True
        return res

