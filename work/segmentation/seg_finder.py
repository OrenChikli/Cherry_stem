import logging
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb  # , slic, quickshift, watershed
from Hawkeye.src.hawkeye.utils.file_utils import FileUtils
from Hawkeye.src.hawkeye.cv.image import Image


logger = logging.getLogger(__name__)


class SegmentFinder:

    SCALE_TRACKBAR_NAME = 'Scale'
    SIGMA_TRACKBAR_NAME = 'Sigma'
    MSIZE_TRACKBAR_NAME = 'Min Size'

    IMAGE_WINDOW_NAME = 'SegmentFinder'

    def __init__(self, path):
        logger.debug(" -> __init__")

        self.path = path
        self.window_name = self.IMAGE_WINDOW_NAME + ' - ' + self.path
        local_path = FileUtils.download_image_with_cache(path)
        self.image = Image(local_path)
        self.image.prepare_for_detection()
        self.float_image = img_as_float(self.image.resized)

        self.segments = None
        self.boundaries = None
        self.segments_count = 0

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

        self.scale = cv2.getTrackbarPos(SegmentFinder.SCALE_TRACKBAR_NAME, self.window_name)
        self.sigma = cv2.getTrackbarPos(SegmentFinder.SIGMA_TRACKBAR_NAME, self.window_name) / 100
        self.min_size = cv2.getTrackbarPos(SegmentFinder.MSIZE_TRACKBAR_NAME, self.window_name)

        self.segments = felzenszwalb(self.float_image, scale=self.scale, sigma=self.sigma, min_size=self.min_size)
        self.segments_count = len(np.unique(self.segments))

        self.boundaries = mark_boundaries(self.image.resized, self.segments, color=(1, 1, 0))
        cv2.imshow(self.window_name, self.boundaries)

        logger.debug(" <- apply")

    def display(self):
        cv2.imshow(self.window_name, self.image.resized)

        cv2.createTrackbar(SegmentFinder.SCALE_TRACKBAR_NAME, self.window_name, 100, 1000, self.on_trackbar_change)
        cv2.createTrackbar(SegmentFinder.SIGMA_TRACKBAR_NAME, self.window_name, 50, 100, self.on_trackbar_change)
        cv2.createTrackbar(SegmentFinder.MSIZE_TRACKBAR_NAME, self.window_name, 50, 500, self.on_trackbar_change)

        self.apply()


# sf = SegmentFinder('https://images.clarifruit.com/1/2047/2018/08/09/38405-11208.png')
# sf = SegmentFinder('https://images.clarifruit.com/2/2143/2018/11/5/67489-16790.png')
sf = SegmentFinder('https://images.clarifruit.com/2/2143/2018/06/12/28596-46382.png')
# sf = SegmentFinder('/Users/roman/work/ClariFruit/DP/grapes_stem/train2/original/28617-03527.png')

# https://images.clarifruit.com/2/2143/2018/11/18//tmp/70463-27564.png

sf.display()

cv2.waitKey(0)
cv2.destroyAllWindows()
