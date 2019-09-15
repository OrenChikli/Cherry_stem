import logging
import cv2
import numpy as np
# from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb  # , slic, quickshift, watershed
from .utils import Utils
from .common import Common


import matplotlib.pyplot as plt
import os
from work.unet.clarifruit_unet.data_functions import create_path
from tqdm import tqdm
import argparse

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

from datetime import datetime


COLOR_DICT = {'gray':cv2.IMREAD_GRAYSCALE,'color':cv2.IMREAD_UNCHANGED}

logger = logging.getLogger(__name__)


class Segmentation:

    def __init__(self, image,
                 scale=100, sigma=0.5, min_size=50, threshold=1, pr_threshold=0.05):

        logger.debug(" -> __init__")


        self.image = image
        self.segments = None
        self.boundaries = None
        self.segments_count = 0
        self.segments_hsv = None
        self.bg_segments = set()
        self.segments_no_bg = set()
        self.filtered_segments = None


        self.threshold = threshold
        self.pr_threshold = pr_threshold
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size


        logger.debug(" <- __init__")

    def get_segments(self):
        float_image = img_as_float(self.image.resized)
        return felzenszwalb(float_image, scale=self.scale, sigma=self.sigma, min_size=self.min_size)




    def apply_segmentation(self, display_flag=False):

        self.segments = self.get_segments()
        segments = np.unique(self.segments)
        self.segments_count = len(segments)


        if self.image.mask_path is not None:
            self.filter_segments()

        if display_flag:
            cv2.imshow("enhanced mask",self.filtered_segments)
            cv2.waitKey(0)

            #self.boundaries = mark_boundaries(self.image, self.segments, color=(1, 1, 0))
            #plt.imshow(self.boundaries)
            #plt.show()

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


    def segment_iterator(self):
        n_segments = self.segments.max()
        for i in range(n_segments):
            segment = np.where(self.segments == i, True, False)
            yield i, segment

    def filter_segments(self):
        self.filtered_segments = np.zeros_like(self.segments,dtype=np.bool)
        for i, segment in self.segment_iterator():
            seg_sum = np.count_nonzero(segment)
            segment_activation = self.image.mask_resized_binary * segment
            #segment_activation = np.bitwise_and(self.image.mask_resized_binary, segment)
            seg_activation_sum = np.count_nonzero(segment_activation)
            activation_pr = (seg_activation_sum / seg_sum)
            if seg_activation_sum >= self.threshold and activation_pr > self.pr_threshold:
                self.filtered_segments[segment] = True

        #self.filtered_segments *= (255.0/self.filtered_segments.max())
        #self.filtered_segments = self.filtered_segments.astype(np.uint8)



    def mask_color_img(self):
        '''
        img: cv2 image
        mask: bool or np.where
        color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
        alpha: float [0, 1].

        Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
        '''
        out = self.image.resized.copy()
        color = (255, 255, 0)
        alpha = 1
        img_layer = out.copy()
        img_layer[self.filtered_segments] = color
        out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
        return out

    # saving functions
    def save_segments(self, save_path):
        for i, segment in self.segment_iterator():
            seg_name = os.path.join(save_path, f"segment_{i}.jpg")
            segment_image = binary_to_grayscale(segment)
            cv2.imwrite(seg_name, segment_image)


    def save_settings(self, save_path):
        settings_dict = {'threshold': self.threshold,
                         'pr_threshold': self.pr_threshold,
                         'scale': self.scale,
                         'sigma': self.sigma,
                         'min_size': self.min_size}


        settings_path = os.path.join(save_path, 'seg_parameters.txt')

        with open(settings_path, 'w') as f:
            for param_name, param_value in settings_dict.items():
                f.write(f"{param_name} = {param_value}\n")

    @staticmethod
    def create_save_paths(image_name,src_save_path, seg_folder=None, activation_folder=None):
        save_path = create_path(src_save_path, image_name)

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = create_path(save_path,current_time)

        ret = [save_path]
        if seg_folder:
            segments_path = create_path(save_path, seg_folder)
            ret.append(segments_path)
        if activation_folder:
            activation_path = create_path(save_path, activation_folder)
            ret.append(activation_path)

        return ret

    def return_modified_mask(self):
        return self.binary_to_grayscale(self.filtered_segments)



    def save_results(self,src_save_path, seg_folder, activation_folder):

        save_path,segments_path,activation_path = self.create_save_paths(self.image.image_name,
                                                                         src_save_path,
                                                                         seg_folder,
                                                                         activation_folder)

        self.save_settings(save_path)

        #self.save_segments(segments_path)

        res_mask = self.return_modified_mask()
        res_mask_path = os.path.join(activation_path,self.image.image_name)

        res_mask_ontop = self.mask_color_img()
        res_mask_ontop_path = os.path.join(activation_path, f"ontop_{self.image.image_name}")

        cv2.imwrite(res_mask_path, res_mask)
        cv2.imwrite(res_mask_ontop_path, res_mask_ontop)

    @staticmethod
    def binary_to_grayscale(img):
        res = img.copy()
        res = (255 * res).astype(np.uint8)
        return res




def check_folder_path(src_path):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"could find path {src_path}")


def binary_to_grayscale(img):
    res = img.copy()
    res = (255 * res).astype(np.uint8)
    return res





def color_segmentation(img_path):
    nemo = cv2.imread(img_path)
    plt.imshow(nemo)
    plt.show()



def visualize_rgb(img):
    r, g, b = cv2.split(img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = get_pixel_colors(img)
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()


def get_pixel_colors(img):
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    return pixel_colors


def visualize_hsv(img):
    hsv= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = get_pixel_colors(img)
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

def color_thres(img,left_thres,right_thes):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, left_thres,right_thes)
    result = cv2.bitwise_and(img, img, mask=mask)
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()


def color():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-c", "--clusters", required=True, type=int,
                    help="# of clusters")
    args = vars(ap.parse_args())

    # load the image and convert it from BGR to RGB so that
    # we can dispaly it with matplotlib
    image = cv2.imread(args["image"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # show our image
    plt.figure()
    plt.axis("off")
    plt.imshow(image)
