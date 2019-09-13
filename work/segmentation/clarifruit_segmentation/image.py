import logging
import numpy as np
import cv2

from .exceptions import ReadImageException
from .utils import Utils
from .segmentation import Segmentation
from skimage.util import img_as_float

logger = logging.getLogger(__name__)
import os
import shutil


class Image:
    RESIZED_IMAGE_LONG_SIDE = 768

    def __init__(self, img_path, mask_path=None):

        self.image_name = os.path.basename(img_path)
        self.img_path = img_path
        self.mask_path = mask_path

        self.original = None
        self.original_mask = None

        self.debug = None
        self.resized = None
        self.mask_resized = None
        self.resize_factor = None
        #self.hsv = None
        #self.hls = None
        #self.gray = None
        #self.blurred = None
        self.segmentation = None

        self.read_local()

    def move_to(self, dest_path_image, dest_path_label):
        img_path = os.path.join(dest_path_image, self.image_name)
        label_path = os.path.join(dest_path_label, self.image_name)

        _ = shutil.move(self.img_path, img_path)
        _ = shutil.move(self.mask_path, label_path)


    def read_local(self):

        logger.debug(" -> read")
        logger.debug("Reading image %s", self.img_path)

        try:
            logger.debug("Reading image locally by OpenCV")
            self.original = cv2.imread(self.img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        except:
            logger.exception('Failed reading image: ' + self.img_path)
            raise ReadImageException('Failed reading image: ' + self.img_path)

        if self.mask_path:
            logger.debug("Reading mask %s", self.mask_path)

            try:
                logger.debug("Reading mask image locally by OpenCV")
                self.mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)

            except:
                logger.exception('Failed reading  mask image: ' + self.mask_path)
                raise ReadImageException('Failed reading  mask image: ' + self.mask_path)

        if self.original is None:
            error_message = "Can't read image from "
            logger.error(error_message, self.img_path)
            # print(error_message, self.img_path)
            raise ReadImageException(error_message, self.img_path)

        logger.debug(" <- read")

    @staticmethod
    def _convert_to_bgr(image):

        new_image = None

        # Checking the number of channels in the read image.
        if image.shape[-1] == 4:
            logger.debug("Converting image from RGBA to BGR")
            new_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            # new_image = self.convert_to_jpg(self.original)
        elif image.shape[-1] == 3:
            logger.debug("Converting image from RGB to BGR")
            new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # new_image = self.convert_to_jpg(self.original)
        elif image.shape[-1] == 1:
            logger.debug("Converting image from GRAY to BGR")
            new_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            logger.debug("Unknown image format")
            # new_image = np.repeat(image[:,:,None], 3, 2).astype(np.uint8)
            # new_image = cv2.cvtColor((image / 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return new_image

    @staticmethod
    def convert_to_jpg(img):
        logger.debug('Converting to .jpg')
        retval, buffer = cv2.imencode('.jpg', img)
        img_jpg = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return img_jpg

    def prepare_for_detection(self):

        logger.debug(" -> prepare_for_detection")

        self.resized, self.resize_factor = Utils.resize_image(self.original, self.RESIZED_IMAGE_LONG_SIDE)

        # self.resized1, self.resize_factor = Utils.resize_image(self.original, self.RESIZED_IMAGE_LONG_SIDE)
        # self.resized = self.white_balance(self.resized1)

        #self.debug = self.resized.copy()
        #self.gray = cv2.cvtColor(self.resized, cv2.COLOR_BGR2GRAY)
        #self.blurred = cv2.GaussianBlur(self.gray, (5, 5), 2, 2)
        #self.hsv = cv2.cvtColor(self.resized, cv2.COLOR_BGR2HSV)
        #self.hls = cv2.cvtColor(self.resized, cv2.COLOR_BGR2HLS)
        self.float = img_as_float(self.original)

        if self.mask is not None:
            self.mask_resized, _ = Utils.resize_image(self.mask,self.RESIZED_IMAGE_LONG_SIDE)
            self.mask_resized_binary = np.where(self.mask_resized == 255, True, False)

        # self.hsv[:, :, 2] = np.clip(self.hsv[:, :, 2].astype(np.uint16) + 24, 0, 255).astype(np.uint8)
        # self.resized = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)


        logger.debug(" <- prepare_for_detection")


    def adjust_gamma(self, image, gamma=1.0):

        inv_gamma = 1.0 / gamma
        # table1 = np.array([((i / 255.0) ** inv_gamma) * 255
        #                   for i in np.arange(0, 256)]).astype("uint8")

        table = (((np.arange(0, 256) / 255.) ** inv_gamma) * 255).astype(np.uint8)

        return cv2.LUT(image, table)

    # virtual
    def get_segmentation_class(self):
        return Segmentation(self)

    def get_hsv_of_original_image(self):
        return cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)

    def get_rgb_of_original_image(self):
        return cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)

    def convert_to_original_size(self, value):
        return value / self.resize_factor

    def convert_to_resized_size(self, value):
        return value * self.resize_factor
