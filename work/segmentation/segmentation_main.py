from work.annotation import fruits_anno
from work.unet.data_functions import *

from work.segmentation.clarifruit_segmentation import segmentation,seg_filter,seg_finder,seg_info
from work.segmentation.clarifruit_segmentation.image import Image
import numpy as np
from tqdm import tqdm

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
    img = Image(img_path)
    #img = cv2.imread(img_path, COLOR_DICT[img_color])
    mask = cv2.imread(mask_imgh_path, cv2.IMREAD_GRAYSCALE)
    mask_binary = np.where(mask == 255, True, False)  # create binary version of the mask image

    # segmentation enhancment
    sg = segmentation.Segmentation(image=img.original, ground_truth=mask_binary)
    sg.apply_segmentation(scale=scale,
                          sigma=sigma,
                          min_size=min_size,
                          display_flag=boundaries_display_flag)

    seg_activation = sg.filter_segments(threshold=threshold)
    curr_activation_full = os.path.join(curr_activation_path, f'thres_{threshold}.jpg')

    # show on source_image

    weighted = segmentation.mask_color_img(img.original, seg_activation, draw_color, draw_alpha)

    seg_out_path_final = os.path.join(curr_activation_path, f'thres_{threshold}_weighted.jpg')

    if save_flag:
        sg.save_segments(curr_segments_path)
        cv2.imwrite(curr_activation_full, segmentation.binary_to_grayscale(seg_activation))
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
        save_segment = segmentation.binary_to_grayscale(curr_segment)
        cv2.imwrite(save_path, save_segment)



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