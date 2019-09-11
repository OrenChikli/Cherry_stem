import logging
import numpy as np
import cv2
#from .aws.s3 import S3
from work.segmentation.exceptions import ReadImageException
from work.segmentation.utils import Utils
from work.segmentation.segmentation import Segmentation

logger = logging.getLogger(__name__)


class Image:
    RESIZED_IMAGE_LONG_SIDE = 768

    def __init__(self, img_path):
        self.img_path = img_path

        self.original = None
        self.debug = None
        self.resized = None
        self.resize_factor = None
        self.hsv = None
        self.hls = None
        self.gray = None
        self.blurred = None
        self.segmentation = None

        # self.lab = None
        # self.ycc = None

        #self.read()
        self.read_local()

    """    
    def read(self):

        logger.debug(" -> read")
        logger.debug("Reading image %s", self.img_path)

        s3_images_bucket_name = 'food-meter-images'

        try:
            url_parse_result = urlparse(self.img_path)
            if url_parse_result.scheme in ['https', 'http', 's3']:
                logger.debug("Remote image path found")

                path = url_parse_result.path[1:]  # removing leading '/'
                if s3_images_bucket_name in path:
                    # Removing the first element of the path (assuming the bucket name may only appear in the first element)
                    logger.debug("Bucket name found in path (%s), removing it.", s3_images_bucket_name)
                    path = str(pathlib.Path(*pathlib.Path(path).parts[1:]))

                # Getting image from s3
                logger.debug('Trying to download the image from s3, %s', path)
                s3 = S3(s3_images_bucket_name)
                img_file_content = s3.get_file_content(path)

                if img_file_content is not None:
                    logger.debug('Image downloaded from s3. Opening in using PIL')
                    # Image downloaded from s3
                    # img_arr = np.frombuffer(img_file_content, np.uint8)
                    # self.original = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                    image = np.array(pil_image.open(io.BytesIO(img_file_content)), dtype=np.uint8)
                else:
                    # Image not found in s3, download from the url
                    logger.debug('Image not found in s3, opening from remote location using skimage.')
                    image = sk_io.imread(self.img_path)

                self.original = self._convert_to_bgr(image)
            else:
                # Local path
                logger.debug("Reading image locally by OpenCV")
                self.original = cv2.imread(self.img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                # self.original = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
                # cv2.IMREAD_IGNORE_ORIENTATION

            # self.original = self._convert_to_bgr(image)
        except:
            logger.exception('Failed reading image: ' + self.img_path)
            raise ReadImageException('Failed reading image: ' + self.img_path)

        if self.original is None:
            error_message = "Can't read image from "
            logger.error(error_message, self.img_path)
            # print(error_message, self.img_path)
            raise ReadImageException(error_message, self.img_path)

        logger.debug(" <- read")"""


    def read_local(self):

        logger.debug(" -> read")
        logger.debug("Reading image %s", self.img_path)

        try:
            logger.debug("Reading image locally by OpenCV")
            self.original = cv2.imread(self.img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        except:
            logger.exception('Failed reading image: ' + self.img_path)
            raise ReadImageException('Failed reading image: ' + self.img_path)

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

        self.debug = self.resized.copy()
        self.gray = cv2.cvtColor(self.resized, cv2.COLOR_BGR2GRAY)
        self.blurred = cv2.GaussianBlur(self.gray, (5, 5), 2, 2)
        self.hsv = cv2.cvtColor(self.resized, cv2.COLOR_BGR2HSV)
        self.hls = cv2.cvtColor(self.resized, cv2.COLOR_BGR2HLS)

        # self.hsv[:, :, 2] = np.clip(self.hsv[:, :, 2].astype(np.uint16) + 24, 0, 255).astype(np.uint8)
        # self.resized = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

        # self.lab = cv2.cvtColor(self.resized, cv2.COLOR_BGR2LAB)
        # self.ycc = cv2.cvtColor(self.resized, cv2.COLOR_BGR2YCR_CB)

        self.segmentation = self.get_segmentation_class()
        self.segmentation.apply()

        logger.debug(" <- prepare_for_detection")

    # def adjust_white_balance_by_ref_object(self, ref_object):
    #     median_color_bgr = ref_object.get_median_color()
    #
    #     gamma = np.log(255.) / np.log(median_color_bgr)
    #
    #     b = self.adjust_gamma(self.resized[:, :, 0], gamma[0])
    #     g = self.adjust_gamma(self.resized[:, :, 1], gamma[1])
    #     r = self.adjust_gamma(self.resized[:, :, 2], gamma[2])
    #
    #     result = np.stack((b, g, r), axis=-1)
    #
    #     if Common.imgLogLevel in ['trace']:
    #         cv2.imshow("gamma", result)
    #
    #     median_color_bgr1 = ref_object.get_median_color(result, 1)
    #
    #     x = 0

    # def white_balance(self, img):
    #     result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #     avg_a = np.average(result[:, :, 1])
    #     avg_b = np.average(result[:, :, 2])
    #     result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    #     result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    #     result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    #     return result

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
