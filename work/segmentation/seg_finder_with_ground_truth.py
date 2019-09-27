import logging
import cv2

from auxiliary.image import Image
from .segmentation import Segmentation


logger = logging.getLogger(__name__)


class MaskSegmentFinder:
    PR_THRESHOLD_TRACKBAR_NAME = 'pr_threshold'
    THRESHOLD_TRACKBAR_NAME = 'threshold'
    SCALE_TRACKBAR_NAME = 'Scale'
    SIGMA_TRACKBAR_NAME = 'Sigma'
    MSIZE_TRACKBAR_NAME = 'Min Size'

    IMAGE_WINDOW_NAME = 'SegmentFinder'

    def __init__(self, path,mask_path=None):
        logger.debug(" -> __init__")

        self.path = path
        self.window_name = self.IMAGE_WINDOW_NAME + ' - ' + self.path

        self.image = Image(path,mask_path)
        self.image.prepare_for_detection()

        self.segmentation = Segmentation(self.image)

        logger.debug(" <- __init__")

    def on_trackbar_change(self, x):
        logger.debug('Trackbar - x: %d', x)
        self.apply()

    def apply(self):

        logger.debug(" -> apply")

        self.segmentation.pr_threshold = cv2.getTrackbarPos(MaskSegmentFinder.PR_THRESHOLD_TRACKBAR_NAME, self.window_name) / 100
        self.segmentation.threshold = cv2.getTrackbarPos(MaskSegmentFinder.THRESHOLD_TRACKBAR_NAME, self.window_name)
        self.segmentation.scale = cv2.getTrackbarPos(MaskSegmentFinder.SCALE_TRACKBAR_NAME, self.window_name)
        self.segmentation.sigma = cv2.getTrackbarPos(MaskSegmentFinder.SIGMA_TRACKBAR_NAME, self.window_name) / 100
        self.segmentation.min_size = cv2.getTrackbarPos(MaskSegmentFinder.MSIZE_TRACKBAR_NAME, self.window_name)

        self.segmentation.apply_segmentation()

        cv2.imshow(self.window_name, self.segmentation.mask_color_img())

        logger.debug(" <- apply")

    def display(self):
        cv2.imshow(self.window_name, self.image.resized)

        cv2.createTrackbar(MaskSegmentFinder.PR_THRESHOLD_TRACKBAR_NAME, self.window_name,1, 100, self.on_trackbar_change)
        cv2.createTrackbar(MaskSegmentFinder.THRESHOLD_TRACKBAR_NAME,self.window_name,1,100, self.on_trackbar_change)
        cv2.createTrackbar(MaskSegmentFinder.SCALE_TRACKBAR_NAME, self.window_name, 100, 1000, self.on_trackbar_change)
        cv2.createTrackbar(MaskSegmentFinder.SIGMA_TRACKBAR_NAME, self.window_name, 50, 100, self.on_trackbar_change)
        cv2.createTrackbar(MaskSegmentFinder.MSIZE_TRACKBAR_NAME, self.window_name, 5, 50, self.on_trackbar_change)

        self.apply()



