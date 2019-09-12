import logging
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb  # , slic, quickshift, watershed
#from Hawkeye.src.hawkeye.utils.file_utils import FileUtils
#from Hawkeye.src.hawkeye.cv.image import Image
from .image import Image
from .segmentation import Segmentation


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
        #self.float_image = img_as_float(self.image.resized)

        self.segmentation = Segmentation(self.image)

        self.filtered_segments = None
        self.disp_mask = None

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

        self.segmentation.apply_segmentation(scale=self.scale,sigma=self.sigma,min_size=self.min_size)

        self.segmentation.filter_segments(self.threshold)
        #self.disp_mask= Segmentation.binary_to_grayscale(self.segmentation.filtered_segments)
        self.disp_mask = self.segmentation.mask_color_img(self)

        #self.boundaries = mark_boundaries(self.image.resized, self.filtered_segments, color=(1, 1, 0))
        cv2.imshow(self.window_name, self.disp_mask)

        logger.debug(" <- apply")

    def display(self):
        cv2.imshow(self.window_name, self.image.resized)
        cv2.createTrackbar(MaskSegmentFinder.THRESHOLD_TRACKBAR_NAME,self.window_name,1,100, self.on_trackbar_change)
        cv2.createTrackbar(MaskSegmentFinder.SCALE_TRACKBAR_NAME, self.window_name, 100, 1000, self.on_trackbar_change)
        cv2.createTrackbar(MaskSegmentFinder.SIGMA_TRACKBAR_NAME, self.window_name, 50, 100, self.on_trackbar_change)
        cv2.createTrackbar(MaskSegmentFinder.MSIZE_TRACKBAR_NAME, self.window_name, 50, 500, self.on_trackbar_change)

        self.apply()



