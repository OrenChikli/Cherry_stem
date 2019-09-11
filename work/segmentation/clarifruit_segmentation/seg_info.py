import logging
import numpy as np
import cv2
#from Hawkeye.src.hawkeye.cv.image import Image
from .image import Image
#from Hawkeye.src.hawkeye.utils.file_utils import FileUtils

logger = logging.getLogger(__name__)

# build_info = cv2.getBuildInformation()


class SegmentViewer:

    HUE_TRACKBAR_NAME = 'Hue'
    SATURATION_TRACKBAR_NAME = 'Saturation'
    VALUE_TRACKBAR_NAME = 'Value'

    IMAGE_WINDOW_NAME = 'SegmentViewer'

    def __init__(self, path):
        # image = cv2.imread("/Users/roman/work/Acclaro/Temp/images.clarifruit.com/2/2143/2018/10/2/57941-64146.png")
        # image = Image('https://images.clarifruit.com/26/3130/2018/10/09/60488-49030.png')

        self.path = path
        self.window_name = self.IMAGE_WINDOW_NAME + ' - ' + self.path
        #local_path = FileUtils.download_image_with_cache(path)
        self.image = Image(path)
        self.image.prepare_for_detection()

    def on_trackbar_change(self, x):
        logger.debug('Trackbar - x: %d', x)

    def display_sigmentation_with_info(self):

        # image = Image('https://images.clarifruit.com/26/3130/2018/10/09/60488-49030.png')
        # image.prepare_for_detection()

        # image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

        # cv2.imshow("image", image.resized)
        # cv2.imshow(self.window_name, self.image.segmentation.get_boundaries())

        height, width = self.image.resized.shape[:2]
        if height > width:
            cv2.imshow(self.window_name, self.image.segmentation.get_boundaries().transpose(1, 0, 2))
        else:
            cv2.imshow(self.window_name, self.image.segmentation.get_boundaries())

        # height, width = self.image.resized.shape[:2]
        # if height > width:
        #     cv2.imshow('Original', self.image.resized.transpose(1, 0, 2))
        #     cv2.imshow(self.window_name, self.image.resized.transpose(1, 0, 2))
        # else:
        #     cv2.imshow('Original', self.image.resized)
        #     cv2.imshow(self.window_name, self.image.resized)


        cv2.setMouseCallback(self.window_name, self.on_click)
        # create trackbars for color change
        cv2.createTrackbar(SegmentViewer.HUE_TRACKBAR_NAME, self.window_name, 0, 180, self.on_trackbar_change)
        cv2.createTrackbar(SegmentViewer.SATURATION_TRACKBAR_NAME, self.window_name, 0, 255, self.on_trackbar_change)
        cv2.createTrackbar(SegmentViewer.VALUE_TRACKBAR_NAME, self.window_name, 0, 255, self.on_trackbar_change)

        # Requires build with QT support
        # cv2.displayStatusBar(self.window_name, text='123')

    def on_click(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            a = 0

            # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:

            a = x
            x = y
            y = a

            seg_val = self.image.segmentation.segments[y, x]
            seg_h, seg_s, seg_v = self.image.segmentation.get_segment_hsv(seg_val)
            # print(self.image.segmentation.segments == seg_val)

            count = np.count_nonzero(self.image.segmentation.segments == seg_val)

            logger.debug('x: %d, y: %d, size: %d, segment#: %d', x, y, count, seg_val)
            logger.debug('hsv: [%.2f, %.2f, %.2f]', seg_h, seg_s, seg_v)

            cv2.setTrackbarPos(SegmentViewer.HUE_TRACKBAR_NAME, self.window_name, np.round(seg_h).astype(np.uint8))
            cv2.setTrackbarPos(SegmentViewer.SATURATION_TRACKBAR_NAME, self.window_name, np.round(seg_s).astype(np.uint8))
            cv2.setTrackbarPos(SegmentViewer.VALUE_TRACKBAR_NAME, self.window_name, np.round(seg_v).astype(np.uint8))

            # logger.debug('segment #: %d', self.image.segmentation.segments[y, x])