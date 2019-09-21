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

    def __init__(self, img_path, mask_path=None):

        self.image_name = os.path.basename(img_path)
        self.img_path = img_path
        self.img = None

        self.mask_path = mask_path
        self.mask = None

        self.segmentation = None

        self.read_local()

    def move_to(self, dest_path_image, dest_path_label):
        logger.debug("-> move_to")
        logger.debug(f"moving images to {dest_path_image} and maskes to {dest_path_label}")
        img_path = os.path.join(dest_path_image, self.image_name)
        label_path = os.path.join(dest_path_label, self.image_name)

        _ = shutil.move(self.img_path, img_path)
        _ = shutil.move(self.mask_path, label_path)

        logger.debug(" <- move_to")

    def read_local(self):

        logger.debug(" -> read")
        logger.debug("Reading image %s", self.img_path)

        try:
            logger.debug("Reading image locally by OpenCV")
            self.img = cv2.imread(self.img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

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

        if self.img is None:
            error_message = "Can't read image from "
            logger.error(error_message, self.img_path)
            raise ReadImageException(error_message, self.img_path)

        if self.mask_path is not None and self.mask is None:
            error_message = "Can't read mask from "
            logger.error(error_message, self.mask_path)
            raise ReadImageException(error_message, self.mask_path)



        logger.debug(" <- read")

