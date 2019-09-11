import logging
import numpy as np
import cv2
#from Hawkeye.src.hawkeye.cv.image import Image
#from Hawkeye.src.hawkeye.utils.file_utils import FileUtils

from work.segmentation.image import Image

logger = logging.getLogger(__name__)

# build_info = cv2.getBuildInformation()


class SegmentFilter:

    MIN_HUE_TRACKBAR_NAME = 'Min Hue'
    MIN_SATURATION_TRACKBAR_NAME = 'Min Saturation'
    MIN_VALUE_TRACKBAR_NAME = 'Min Value'
    MAX_HUE_TRACKBAR_NAME = 'Max Hue'
    MAX_SATURATION_TRACKBAR_NAME = 'Max Saturation'
    MAX_VALUE_TRACKBAR_NAME = 'Max Value'

    INVERT_HUE_TRACKBAR_NAME = 'Invert Hue Filter \n 0: min < - > max \n 1: Inverted'

    IMAGE_WINDOW_NAME = 'SegmentFilter'

    def __init__(self, path):
        self.path = path
        self.window_name = self.IMAGE_WINDOW_NAME + ' - ' + self.path
        #local_path = FileUtils.download_image_with_cache(path)
        self.image = Image(path)
        self.image.prepare_for_detection()
        self.seg_image = None

    def on_trackbar_change(self, x):
        logger.debug('Trackbar - x: %d', x)
        self.filter_segments()

    def display_sigmentation_filter(self):

        # cv2.imshow(self.window_name, self.image.segmentation.boundaries)

        height, width = self.image.resized.shape[:2]
        if height > width:
            cv2.imshow('Original', self.image.resized.transpose(1, 0, 2))
            cv2.imshow(self.window_name, self.image.resized.transpose(1, 0, 2))
        else:
            cv2.imshow('Original', self.image.resized)
            cv2.imshow(self.window_name, self.image.resized)

        cv2.setMouseCallback(self.window_name, self.on_click)

        # create trackbars for color limits
        cv2.createTrackbar(SegmentFilter.MIN_HUE_TRACKBAR_NAME, self.window_name, 40, 180, self.on_trackbar_change)
        cv2.createTrackbar(SegmentFilter.MAX_HUE_TRACKBAR_NAME, self.window_name, 165, 180, self.on_trackbar_change)
        cv2.createTrackbar(SegmentFilter.INVERT_HUE_TRACKBAR_NAME, self.window_name, 1, 1, self.on_trackbar_change)

        cv2.createTrackbar(SegmentFilter.MIN_SATURATION_TRACKBAR_NAME, self.window_name, 70, 255, self.on_trackbar_change)
        cv2.createTrackbar(SegmentFilter.MAX_SATURATION_TRACKBAR_NAME, self.window_name, 255, 255,
                           self.on_trackbar_change)

        cv2.createTrackbar(SegmentFilter.MIN_VALUE_TRACKBAR_NAME, self.window_name, 70, 255, self.on_trackbar_change)
        cv2.createTrackbar(SegmentFilter.MAX_VALUE_TRACKBAR_NAME, self.window_name, 255, 255, self.on_trackbar_change)

        self.filter_segments()

        # Requires build with QT support
        # cv2.displayStatusBar(self.window_name, text='123')

    def on_click(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            a = 0

            # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:

            seg_val = self.image.segmentation.segments[y, x]
            seg_h, seg_s, seg_v = self.image.segmentation.get_segment_hsv(seg_val)
            # print(self.image.segmentation.segments == seg_val)

            count = np.count_nonzero(self.image.segmentation.segments == seg_val)

            logger.debug('x: %d, y: %d, size: %d, segment#: %d', x, y, count, seg_val)
            logger.debug('hsv: [%.2f, %.2f, %.2f]', seg_h, seg_s, seg_v)

            # cv2.setTrackbarPos(SegmentFilter.HUE_TRACKBAR_NAME, self.window_name, np.round(seg_h).astype(np.uint8))
            # cv2.setTrackbarPos(SegmentFilter.SATURATION_TRACKBAR_NAME, self.window_name, np.round(seg_s).astype(np.uint8))
            # cv2.setTrackbarPos(SegmentFilter.VALUE_TRACKBAR_NAME, self.window_name, np.round(seg_v).astype(np.uint8))

            # logger.debug('segment #: %d', self.image.segmentation.segments[y, x])

    def filter_segments(self):

        min_h = cv2.getTrackbarPos(SegmentFilter.MIN_HUE_TRACKBAR_NAME, self.window_name)
        max_h = cv2.getTrackbarPos(SegmentFilter.MAX_HUE_TRACKBAR_NAME, self.window_name)
        inv_h = cv2.getTrackbarPos(SegmentFilter.INVERT_HUE_TRACKBAR_NAME, self.window_name)

        min_s = cv2.getTrackbarPos(SegmentFilter.MIN_SATURATION_TRACKBAR_NAME, self.window_name)
        max_s = cv2.getTrackbarPos(SegmentFilter.MAX_SATURATION_TRACKBAR_NAME, self.window_name)

        min_v = cv2.getTrackbarPos(SegmentFilter.MIN_VALUE_TRACKBAR_NAME, self.window_name)
        max_v = cv2.getTrackbarPos(SegmentFilter.MAX_VALUE_TRACKBAR_NAME, self.window_name)

        def filter_hsv(seg_h, seg_s, seg_v):
            return min_s <= seg_s <= max_s and \
                   min_v <= seg_v <= max_v and \
                   ((inv_h == 1 and (seg_h < min_h or seg_h > max_h)) or
                    (inv_h == 0 and min_h <= seg_h <= max_h))

        # min_h = 60
        # max_h = 120
        # inv_h = 1
        #
        # min_s = 30
        # max_s = 255
        #
        # min_v = 60
        # max_v = 255

        mask, _ = self.image.segmentation.filter_by_hsv(filter_hsv)

        self.seg_image = cv2.bitwise_and(self.image.resized, self.image.resized, mask=mask)

        height, width = self.image.resized.shape[:2]
        if height > width:
            cv2.imshow(self.window_name, self.seg_image.transpose(1, 0, 2))
        else:
            cv2.imshow(self.window_name, self.seg_image)

    def save_with_mask(self):
        logger.debug('Save path: ' + self.image.img_path)
        cv2.imwrite(self.image.img_path + '.m.png', self.image.resized)
        cv2.imwrite(self.image.img_path + '.mm.png', self.seg_image)


