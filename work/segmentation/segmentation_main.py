from work.unet.clarifruit_unet.data_functions import *

from work.segmentation.clarifruit_segmentation import segmentation,seg_filter,seg_finder,seg_info,\
    seg_finder_with_ground_truth

from tqdm import tqdm
from work.segmentation.clarifruit_segmentation.image import Image
from datetime import datetime


def segment(orig_path, mask_path, seg_path, seg_folder, activation_folder,
            threshold=1,pr_threshold=0.05, scale=100, sigma=0.5, min_size=50,
            boundaries_display_flag=False,save_flag=True):

    # load the src image and mask image
    img = Image(orig_path, mask_path)
    img.prepare_for_detection()

    # segmentation enhancment
    sg = segmentation.Segmentation(image=img,
                                   scale=scale,
                                   sigma=sigma,
                                   min_size=min_size,
                                   threshold=threshold,
                                   pr_threshold=pr_threshold)

    sg.apply_segmentation(display_flag=boundaries_display_flag)

    if save_flag:
        seg_path = os.path.join(seg_path, 'individual')
        sg.save_results(seg_path,seg_folder,activation_folder)

    return sg


def segment_multi(orig_path, mask_path, seg_path, seg_folder, seg_activation_folder, img_list,settings_dict):

    dir_save_path = create_path(seg_path, 'final')

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    dir_save_path = create_path(dir_save_path, current_time)

    for img in tqdm(img_list):
        curr_segment = segment(img, orig_path, mask_path, seg_path, seg_folder, seg_activation_folder,
                               threshold=settings_dict['threshold'],
                               pr_threshold=settings_dict['pr_threshold'],
                               scale=settings_dict['scale'],
                               sigma=settings_dict['sigma'],
                               min_size=settings_dict['min_size'],
                               boundaries_display_flag=False,
                               save_flag=False)

        res_mask = curr_segment.return_modified_mask()
        orig_mask = curr_segment.image.mask_resized

        save_path = os.path.join(dir_save_path, img)
        save_path_orig_mask = os.path.join(dir_save_path,f"orig_{img}")
        cv2.imwrite(save_path, res_mask)
        cv2.imwrite(save_path_orig_mask,orig_mask)

    curr_segment.save_settings(dir_save_path)


def use_segment(image_name,orig_path,mask_path,seg_path,settings_dict):

    seg_folder = 'segmentation'
    seg_activation_folder = 'activation'

    img_path = os.path.join(orig_path,image_name)
    img_mask_path = os.path.join(mask_path,image_name)

    _ = segment(img_path, img_mask_path, seg_path, seg_folder, seg_activation_folder,
                           threshold=settings_dict['threshold'],
                           pr_threshold=settings_dict['pr_threshold'],
                           scale=settings_dict['scale'],
                           sigma=settings_dict['sigma'],
                           min_size=settings_dict['min_size'],
                           boundaries_display_flag=False,
                           save_flag=True)


# use roman functions
# TODO finish move images function
""" 
def move_images(orig_path,mask_path,img_list,dest_path):
    dest_image_path = create_path(dest_path,'image')
    dest_mask_path = 
    for img_name in img_list:
        curr_img_path = os.path.join(orig_path,img_name)
        curr_mask_path = os.path.join(mask_path,img_name)
        img = Image(img_name,curr_img_path,curr_mask_path)
        dest_img
        img.move_to()
"""

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

def use_seg_finder_with_ground_truth(img_path,mask_path):
    sf = seg_finder_with_ground_truth.MaskSegmentFinder(img_path,mask_path)

    sf.display()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def use_seg_filter(img_path):
    sf = seg_filter.SegmentFilter(img_path)

    sf.display_sigmentation_filter()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def canny(img):

    img = cv2.imread('messi5.jpg', 0)
    edges = cv2.Canny(img, 100, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()




def seg_main():
    image_name = '74714-32897.png.jpg'

    #orig_path =r'D:\Clarifruit\cherry_stem\data\difficult\image'
    orig_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes\image'
    mask_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes\label'
    seg_path =  r'D:\Clarifruit\cherry_stem\data\segmentation'

    seg_folder='segmentation'
    seg_activation_folder ='activation'

    img_path = os.path.join(orig_path, image_name)
    img_mask_path = os.path.join(mask_path,image_name)

    img_list = [
                '45783-98635.png.jpg',
                '71089-01084.png.jpg',
                '72492-85602.png.jpg',
                '72520-70104.png.jpg',
                '72590-54586.png.jpg',
                '72592-11978.png.jpg',
                '72596-28736.png.jpg',
                '74714-32897.png.jpg',
                '78702-22132.png.jpg',
                '78702-32898.png.jpg',
                '78702-35309.png.jpg',
                '78712-02020.png.jpg']

    settings_dict = {'threshold': 10,
                     'pr_threshold': 0.05,
                     'scale': 100,
                     'sigma': 0.5,
                     'min_size': 50}

    use_segment(image_name,orig_path, mask_path, seg_path, settings_dict)
    #use_seg_info(img_path)
    #use_seg_finder(img_path)
    #use_seg_filter(img_path)
    #use_seg_finder_with_ground_truth(img_path,img_mask_path)
    #img_list = [item.name for item in os.scandir(orig_path)]
    #segment_multi(orig_path, mask_path, seg_path, seg_folder, seg_activation_folder, img_list, settings_dict)

if __name__ == '__main__':
    seg_main()