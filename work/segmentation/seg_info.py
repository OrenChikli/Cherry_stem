import logging
import numpy as np
import cv2
from Hawkeye.src.hawkeye.cv.image import Image
from Hawkeye.src.hawkeye.utils.file_utils import FileUtils

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
        local_path = FileUtils.download_image_with_cache(path)
        self.image = Image(local_path)
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


# sv = SegmentViewer('https://images.clarifruit.com/26/3130/2018/10/09/60488-49030.png')

# Cherries color pallete
# sv = SegmentViewer('/Users/roman/work/ClariFruit/Docs/cheries.jpg')

# Peach color pallete
# sv = SegmentViewer('/Users/roman/work/ClariFruit/Docs/peach.jpg')

# Peach
# sv = SegmentViewer('https://images.clarifruit.com/1/2047/2018/08/09/38405-11208.png')
# sv = SegmentViewer('https://images.clarifruit.com/1/2047/2018/08/09/38405-98680.png')
# sv = SegmentViewer('/Users/roman/work/ClariFruit/DP/tomato_patch/31734-75586_hed.png')

# Cherri
# sv = SegmentViewer('https://images.clarifruit.com/1/2047/2018/08/09/38376-48758.png')

# Blue scarlotta
# sv = SegmentViewer('https://images.clarifruit.com/2/2143/2018/10/29/65926-72275.png')

# Tomato
# /Users/roman/Dropbox/ClariFruit_Lab/tomato/Color Palette/Color Palette.pdf
# sv = SegmentViewer('/Users/roman/Dropbox/ClariFruit_Dev/CV/tomato_pallete.png')

# Black grapes
# sv = SegmentViewer('/Users/roman/work/ClariFruit/DP/grapes_stem/train2/original/28617-03527.png')

# Green grapes
# sv = SegmentViewer('https://images.clarifruit.com/1/2047/2019/01/02/78603-15884.png')

# Green grapes pallete
# sv = SegmentViewer('/Users/roman/work/ClariFruit/Docs/green_grapes.jpg')
# sv = SegmentViewer('/Users/roman/work/ClariFruit/Docs/grapes.jpg')

# Dark grapes pallete
# sv = SegmentViewer('/Users/roman/work/ClariFruit/Docs/dark_grapes.jpg')

# sv = SegmentViewer('https://images.clarifruit.com/2/2143/2018/11/11//tmp/69093-78549.png')

# sv = SegmentViewer('https://images.clarifruit.com/2/2143/2018/11/2/66911-51788.png')

# sv = SegmentViewer('https://images.clarifruit.com/11/1931/2018/06/06/28378-41458.png')

# Grapes - Red Globe, white balance
# /Users/roman/work/ClariFruit/Docs/RedGlobe.png - Pallete
# https://images.clarifruit.com/2/2143/2018/10/31/66536-25575.png
# https://images.clarifruit.com/2/2143/2018/10/29/66038-90436.png
# sv = SegmentViewer('https://images.clarifruit.com/2/2143/2018/10/29/66038-90436.png')

# Tomato palette
# sv = SegmentViewer('/Users/roman/Dropbox/ClariFruit_Lab/tomato/Color Palette/all.png')
# sv = SegmentViewer('/Users/roman/Dropbox/ClariFruit_Lab/tomato/Color Palette/verbree/all.png')

# Cherry tomatoes with stem
# sv = SegmentViewer('/Volumes/Samsung_T5/data/tomatoes_detection/images/74065-72783.png.jpg')

# Cherries
sv = SegmentViewer('https://images.clarifruit.com/8262/1967/2019/02/14/86182-24673.png')




sv.display_sigmentation_with_info()

cv2.waitKey(0)
cv2.destroyAllWindows()
