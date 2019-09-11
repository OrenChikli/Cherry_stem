import logging
import cv2
import numpy as np
# from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb  # , slic, quickshift, watershed
from work.segmentation.utils import Utils
from work.segmentation.common import Common

logger = logging.getLogger(__name__)


class Segmentation:

    def __init__(self, image):

        logger.debug(" -> __init__")

        self.image = image
        self.segments = None
        self.boundaries = None
        self.segments_count = 0
        self.segments_hsv = None
        self.bg_segments = set()
        self.segments_no_bg = set()

        logger.debug(" <- __init__")

    def segment_image(self, float_image):
        return felzenszwalb(float_image, scale=100, sigma=0.5, min_size=50)
        # return felzenszwalb(float_image, scale=1, sigma=0.8, min_size=20, multichannel=True)
        # return felzenszwalb(float_image, scale=500, sigma=0.4, min_size=100)

    def apply(self):

        logger.debug(" -> apply")

        logger.debug("Segmenting image...")
        image = self.image.resized
        self.segments = self.segment_image(img_as_float(image))

        segments = np.unique(self.segments)
        self.segments_count = len(segments)
        logger.debug("Number of segments found: %d", self.segments_count)

        self.bg_segments = self.find_backgroud_segments()
        self.segments_no_bg = set(segments) - self.bg_segments

        logger.debug("Calculating segments' hsv...")
        self.segments_hsv = np.array([self._get_segment_hsv(seg_val) for seg_val in range(self.segments_count)])
        # self.segments_hsv = [self._get_segment_hsv(seg_val) for seg_val in range(self.segments_count)]

        if Common.imgLogLevel in ['trace']:
            self.boundaries = mark_boundaries(image, self.segments, color=(1, 1, 0))
            cv2.imshow("Segmented", self.boundaries)

        logger.debug(" <- apply")

    def get_boundaries(self):
        if self.boundaries is None:
            self.boundaries = mark_boundaries(self.image.resized, self.segments, color=(1, 1, 0))

        return self.boundaries

    def filter_by_hsv(self, filter_hsv, segments=None):
        logger.debug(" -> filter_by_hsv")

        # image = self.image.resized
        # mask = np.zeros(image.shape[:2], dtype="uint8")
        # image_hues = self.image.hsv[:, :, 0]

        segments_list = segments if segments is not None else self.segments_no_bg

        selected_segments = []
        # loop over the unique segment values
        # for (i, seg_val) in enumerate(np.unique(self.segments)):
        for seg_val in segments_list:

            # seg_gray = self.image.gray[self.segments == seg_val]

            # seg_h, seg_s, seg_v = self.get_segment_hsv(seg_val)
            seg_h, seg_s, seg_v = self.get_segment_hsv(seg_val)

            # return self.check_color_match_fruit(seg_h, seg_s, seg_v)

            # Checking if the color of the segment matches the fruit
            # if fruit.check_segment_match_fruit(seg_hsv_values):
            # if fruit.check_color_match_fruit(seg_h, seg_s, seg_v):
            if filter_hsv(seg_h, seg_s, seg_v):
                selected_segments.append(seg_val)

                # mask[self.segments == seg_val] = 255
                # seg_values.append(seg_val)

        # boundaries_mask = np.zeros(image.shape[:2], dtype="uint8")

        # boundaries_mask = mark_boundaries(boundaries_mask, self.segments, color=(1, 1, 1))[:, :, 0]
        # boundaries_mask_temp = cv2.bitwise_not(boundaries_mask_temp, mask)
        # boundaries_mask = 1 - boundaries_mask

        # boundaries_mask_gray = cv2.cvtColor(boundaries_mask_temp, cv2.COLOR_BGR2GRAY)

        # mask1 = np.zeros(image.shape[:2], dtype="uint8")
        # mask1[mask1 == [255, 255, 255]] = 255

        # seg_image = cv2.bitwise_and(image, image, mask=cv2.cvtColor(boundaries_mask_temp, cv2.COLOR_BGR2GRAY))

        # bool_arr = np.isin(self.segments, selected_segments)
        mask = np.isin(self.segments, selected_segments).astype(np.uint8) * 255

        # mask[np.isin(self.segments, selected_segments)] = 255

        # mask = bool_arr

        # if Common.imgLogLevel in ['trace']:
        #     # show the masked region
        #     cv2.imshow("Mask", mask)
        #
        #     image = self.image.resized
        #     seg_image = cv2.bitwise_and(image, image, mask=mask)
        #     # seg_image = cv2.bitwise_and(seg_image, seg_image, mask=img_as_ubyte(boundaries_mask))
        #     cv2.imshow("Applied", seg_image)

        logger.debug(" <- filter_by_hsv")

        return mask, set(selected_segments)

    def _get_segment_hsv(self, seg_val):
        # Getting hsv values of the current segment
        seg_hsv_values = self.image.hsv[self.segments == seg_val]
        # seg_hsv_values = self.image.hls[self.segments == seg_val]
        seg_h, seg_h_var = Utils.calc_hue_mean_and_var(seg_hsv_values[:, 0])
        [_, seg_s, seg_v] = seg_hsv_values.mean(axis=0)
        # [_, seg_s_var, seg_v_var] = seg_hsv.var(axis=0)

        return np.array([seg_h, seg_s, seg_v], np.float64)
        # return [seg_h, seg_s, seg_v]

    def get_segment_hsv(self, seg_val):
        return self.segments_hsv[seg_val]

    def filter_segments(self, filter_segment):
        logger.debug(" -> filter_segments")

        # selected_segments = []
        # loop over the unique segment values
        # for seg_val in range(self.segments_count):
        #     if filter_segment(self.segments == seg_val):
        #         selected_segments.append(seg_val)

        selected_segments = [seg_val for seg_val in self.segments_no_bg if filter_segment(self.segments == seg_val)]
        mask = np.isin(self.segments, selected_segments).astype(np.uint8) * 255

        logger.debug(" <- filter_segments")
        return mask, set(selected_segments)

    def find_backgroud_segments(self):

        # For now, consider as background only the segments that touches both sides of the images (vertically or horizontally)
        return (set(self.segments[0, :]) & set(self.segments[-1, :])) | \
               (set(self.segments[:, 0]) & set(self.segments[:, -1]))

    #
    # def filter_by_area(self, area):
    #     for seg_val in self.segments_no_bg:
    #         pass
    #
    # def segment_area(self, seg_id):
    #     pass
    #
    #
