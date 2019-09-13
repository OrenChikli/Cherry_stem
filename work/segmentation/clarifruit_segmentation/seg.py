import cv2
import numpy as np
# from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb  # , slic, quickshift, watershed
import matplotlib.pyplot as plt
import os
from work.unet.clarifruit_unet.data_functions import create_path
from tqdm import tqdm
import argparse

from matplotlib import colors


COLOR_DICT = {'gray':cv2.IMREAD_GRAYSCALE,'color':cv2.IMREAD_UNCHANGED}

class Segmentation:

    def __init__(self, image, ground_truth):

        self.ground_truth = ground_truth
        self.image = image
        self.segments = None
        self.boundaries = None
        self.segments_count = 0
        self.segments_hsv = None

    def get_segments(self, scale=100, sigma=0.5, min_size=50):
        float_image = img_as_float((self.image))
        return felzenszwalb(float_image, scale=scale, sigma=sigma, min_size=min_size)

    def apply_segmentation(self, scale=100, sigma=0.5, min_size=50, display_flag=False):

        self.segments = self.get_segments(scale=scale, sigma=sigma, min_size=min_size)
        segments = np.unique(self.segments)
        self.segments_count = len(segments)

        if display_flag:
            self.boundaries = mark_boundaries(self.image, self.segments, color=(1, 1, 0))
            plt.imshow(self.boundaries)
            plt.show()

    def get_boundaries(self):
        if self.boundaries is None:
            self.boundaries = mark_boundaries(self.image.resized, self.segments, color=(1, 1, 0))

        return self.boundaries

    def save_segments(self, save_path):
        for i, segment in self.segment_iterator():
            seg_name = os.path.join(save_path, f"segment_{i}.jpg")
            segment_image = binary_to_grayscale(segment)
            cv2.imwrite(seg_name, segment_image)

    def segment_iterator(self):
        n_segments = self.segments.max()
        for i in range(n_segments):
            segment = np.where(self.segments == i, True, False)
            yield i, segment

    def filter_segments(self, threshold=1):
        res = np.zeros_like(self.segments, dtype=np.bool)
        for i, segment in self.segment_iterator():
            segment_activation = self.ground_truth * segment
            seg_sum = np.count_nonzero(segment_activation)
            if seg_sum >= threshold:
                res[segment] = True
        return res




def check_folder_path(src_path):
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"could find path {src_path}")


def binary_to_grayscale(img):
    res = img.copy()
    res = (255 * res).astype(np.uint8)
    return res


def mask_color_img(img, mask, color=(0, 255, 255), alpha=0.3):
    '''
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
    alpha: float [0, 1].

    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask] = color
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    return out


def color_segmentation(img_path):
    nemo = cv2.imread(img_path)
    plt.imshow(nemo)
    plt.show()


def segment(image_name, orig_path, mask_path, seg_path, seg_folder, seg_activation_folder,
            threshold=1, scale=100, sigma=0.5, min_size=50,
            draw_color=(255, 0, 255), draw_alpha=1.0,
            boundaries_display_flag=False,
            save_flag=True,
            img_color='color'):
    # segmentaion paths
    seg_path = os.path.join(seg_path, 'individual')
    if save_flag:
        curr_seg_path = create_path(seg_path, image_name)

        curr_segments_path = create_path(curr_seg_path, seg_folder)
        curr_activation_path = create_path(curr_seg_path, seg_activation_folder)
    else:
        curr_seg_path = ""
        curr_segments_path= ""
        curr_activation_path = ""

    # load the src image and mask image
    img_path = os.path.join(orig_path, image_name)
    mask_imgh_path = os.path.join(mask_path, image_name)
    img = cv2.imread(img_path, COLOR_DICT[img_color])
    mask = cv2.imread(mask_imgh_path, cv2.IMREAD_GRAYSCALE)
    mask_binary = np.where(mask == 255, True, False)  # create binary version of the mask image

    # segmentation enhancment
    sg = Segmentation(image=img, ground_truth=mask_binary)
    sg.apply_segmentation(scale=scale,
                          sigma=sigma,
                          min_size=min_size,
                          display_flag=boundaries_display_flag)

    seg_activation = sg.filter_segments(threshold=threshold)
    curr_activation_full = os.path.join(curr_activation_path, f'thres_{threshold}.jpg')

    # show on source_image

    weighted = mask_color_img(img, seg_activation, draw_color, draw_alpha)

    seg_out_path_final = os.path.join(curr_activation_path, f'thres_{threshold}_weighted.jpg')

    if save_flag:
        sg.save_segments(curr_segments_path)
        cv2.imwrite(curr_activation_full, binary_to_grayscale(seg_activation))
        cv2.imwrite(seg_out_path_final, weighted)

    return seg_activation


def segment_multi(orig_path, mask_path, seg_path, seg_folder, seg_activation_folder,img_list,
                  threshold=1, scale=100, sigma=0.5, min_size=50):
    #img_list = os.scandir(orig_path)
    for img in tqdm(img_list):
        curr_segment = segment(img, orig_path, mask_path, seg_path, seg_folder, seg_activation_folder,
                               threshold=threshold, scale=scale, sigma=sigma, min_size=min_size,
                               boundaries_display_flag=False,
                               save_flag=False)
        save_path = os.path.join(seg_path, 'final')
        save_path = os.path.join(save_path, img)
        save_segment = binary_to_grayscale(curr_segment)
        cv2.imwrite(save_path, save_segment)

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