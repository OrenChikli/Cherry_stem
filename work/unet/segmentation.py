import logging
import cv2
import numpy as np
# from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb  # , slic, quickshift, watershed
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)


class Segmentation:

    def __init__(self, image,ground_truth):


        self.ground_truth = ground_truth
        self.image = image
        self.segments = None
        self.boundaries = None
        self.segments_count = 0
        self.segments_hsv = None
        self.bg_segments = set()
        self.segments_no_bg = set()



    def segment_image(self):
        float_image = img_as_float((self.image))
        return felzenszwalb(float_image, scale=100, sigma=0.5, min_size=50)

    def apply(self,display_flag=False):

        self.segments = self.segment_image()
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

    def filter_segments_by_prediction(self, seg_image_bool,num,count_thres=10):

        seg_size = np.count_nonzero(seg_image_bool)
        x = (255*seg_image_bool).astype(np.uint8)
        cv2.imwrite(f'D:\Clarifruit\cherry_stem\data\segmentation\mask_{num}.jpg',x)
        #plt.imshow(x,cmap='gray')
        #plt.show()
        #cv2.imshow('seg', x)
        #cv2.waitKey(1)
        stem_seg = np.where(seg_image_bool,self.ground_truth,0)

        plt.imshow(stem_seg)
        #cv2.waitKey(0)
        stem_seg_sum = np.sum(stem_seg)
        #cv2.waitKey(1)


        #print(f'sum ={stem_seg_sum}')
        #res = stem_seg_sum >= seg_size * count_thres
        res = stem_seg_sum >=count_thres
        #print(res)
        return res


    def save_segments(self,save_path):
        seg_save_path =os.path.join(save_path,'segments')
        for i,segment in self.segment_iterator():

            seg_name = os.path.join(seg_save_path,f"segment_{i}.jpg")
            segment_image = (255 * segment).astype(np.uint8)
            cv2.imwrite(seg_name,segment_image)


    def segment_iterator(self):
        n_segments = self.segments.max()
        for i in range(n_segments):
            segment = np.where(self.segments == i,True,False)
            yield i,segment



    def filter_segments(self, filter_segment=1,threshold=1):
        seg_path = r'D:\Clarifruit\cherry_stem\data\segmentation\74714-32897.png.jpg\activation'
        res_name ='result.jpg'
        x_path = os.path.join(seg_path,res_name)
        res =np.zeros_like(self.segments)
        for i,segment in self.segment_iterator():
            segment_activation = self.ground_truth * segment
            seg_sum = np.count_nonzero(segment_activation)
            if seg_sum >= threshold:
                res[segment] = 255
            #seg_name = os.path.join(seg_path, f"segment_activation_{i}.jpg")np.uint8
        res =res.astype(np.uint8)
        cv2.imwrite(x_path,res)




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