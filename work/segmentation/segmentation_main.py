from work.annotation import fruits_anno
from work.unet.data_functions import *

from work.segmentation import segmentation,seg_filter,seg_finder,seg_info


def activate_segmentation():
    image_name = '45665-81662.png.jpg'

    orig_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\image'
    mask_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\label'
    seg_path = r'D:\Clarifruit\cherry_stem\data\segmentation'

    seg_folder = 'segments'
    seg_activation_folder = 'activation'

    boundaries_display_flag = True
    save_flag = True
    threshold = 50  # for the segmenation folder
    img_color = 'color'  # keep color at the moment doesnt work with grayscale

    # fiz segmentation parameters
    scale = 100
    sigma = 0.5

    min_size = 100

    # mask_draw_params
    color = (255, 0, 255)
    alpha = 1

    segmentation.segment(image_name, orig_path, mask_path, seg_path, seg_folder, seg_activation_folder,
                         threshold, scale, sigma, min_size, color, alpha, boundaries_display_flag, save_flag,
                         img_color)


def get_multi_segments():
    orig_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\image'
    mask_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\label'
    seg_path = r'D:\Clarifruit\cherry_stem\data\segmentation'

    seg_folder = 'segments'
    seg_activation_folder = 'activation'

    difficult_list = ['45665-81662.png.jpg',
                      '45783-98635.png.jpg',
                      '74714-32897.png.jpg',
                      '74714-32897.png.jpg',
                      '74717-45732.png.jpg',
                      '74719-86289.png.jpg',
                      '77824-74792.png.jpg',
                      '78702-22132.png.jpg',
                      '78702-32898.png.jpg',
                      '78702-35309.png.jpg',
                      '78712-02020.png.jpg']

    threshold = 100  # for the segmenation folder

    scale = 100
    sigma = 0.5

    min_size = 100

    segmentation.segment_multi(orig_path, mask_path, seg_path, seg_folder, seg_activation_folder, difficult_list,
                               threshold=threshold, scale=scale, sigma=sigma, min_size=min_size)


def visualize():
    image_name = '45665-81662.png.jpg'

    orig_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\image'

    img_path = os.path.join(orig_path, image_name)
    sv = seg_filter.SegmentFilter(img_path)
    sv.display_sigmentation_filter()

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def hsv():
    image_name = '45665-81662.png.jpg'

    orig_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\image'
    img_color = 'color'

    img_path = os.path.join(orig_path, image_name)
    img = cv2.imread(img_path, segmentation.COLOR_DICT[img_color])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # segmentation.visualize_rgb(img)
    # segmentation.visualize_hsv(img)
    # codes from http://www.workwithcolor.com/green-color-hue-range-01.htm
    # left = (178,236,93) # color Inchworm
    # right = (65,72,51) #Rifle Green
    # segmentation.color_thres(img, left, right)


# use roman functions

def use_seg_info(img_path):

    sv = seg_info.SegmentViewer(img_path)

    sv.display_sigmentation_with_info()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def use_seg_finder(img_path):
    sf = seg_finder.SegmentFinder(img_path)

    sf.display()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def use_seg_filter(img_path):
    sf = seg_filter.SegmentFilter(img_path)

    sf.display_sigmentation_filter()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def seg_main():
    image_name = '45665-81662.png.jpg'

    orig_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\image'
    img_path = os.path.join(orig_path, image_name)

    #use_seg_info(img_path)
    #use_seg_finder(img_path)
    use_seg_filter(img_path)


if __name__ == '__main__':
    seg_main()