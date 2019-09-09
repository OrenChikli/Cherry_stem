
import cv2
import numpy as np
# from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb  # , slic, quickshift, watershed
import matplotlib.pyplot as plt
import os



class Segmentation:

    def __init__(self, image,ground_truth):


        self.ground_truth = ground_truth
        self.image = image
        self.segments = None
        self.boundaries = None
        self.segments_count = 0
        self.segments_hsv = None




    def get_segments(self):
        float_image = img_as_float((self.image))
        #float_image, scale=100, sigma=0.5, min_size=50
        return felzenszwalb(float_image, scale=3.0, sigma=0.95, min_size=5)

    def apply_segmentation(self, display_flag=False):

        self.segments = self.get_segments()
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



    def save_segments(self,save_path):
        for i,segment in self.segment_iterator():

            seg_name = os.path.join(save_path,f"segment_{i}.jpg")
            segment_image = binary_to_grayscale(segment)
            cv2.imwrite(seg_name,segment_image)


    def segment_iterator(self):
        n_segments = self.segments.max()
        for i in range(n_segments):
            segment = np.where(self.segments == i,True,False)
            yield i,segment



    def filter_segments(self,threshold=1):
        res =np.zeros_like(self.segments,dtype=np.bool)
        for i,segment in self.segment_iterator():
            segment_activation = self.ground_truth * segment
            seg_sum = np.count_nonzero(segment_activation)
            if seg_sum >= threshold:
                res[segment] = True
        return res



    def get_modified_mask(self):
        pass



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