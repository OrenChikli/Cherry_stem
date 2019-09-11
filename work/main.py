from work.annotation import fruits_anno
from work.unet.data_functions import *
#from work.unet.model import *
#from work.segmentation.segmentation import *
from work.segmentation import segmentation,seg_filter



def annotate():
    raw_data_path = r'D:\Clarifruit\cherry_stem\data\raw_data'

    ano_path = os.path.join(raw_data_path, 'Cherry_cherry_stem_result.json')
    src_images_path = os.path.join(raw_data_path, 'images_orig')
    csv_path = os.path.join(raw_data_path, 'cherry_stem.csv')

    model_out_path = r'D:\Clarifruit\cherry_stem\data\annotation_output'

    mask_dest_path = os.path.join(model_out_path, 'masks')
    images_dest_path = os.path.join(model_out_path, 'train')

    gl = fruits_anno.GoogleLabels(anno_path=ano_path,
                                  src_images_path=src_images_path,
                                  csv_path=csv_path,
                                  dest_images_path=images_dest_path,
                                  mask_dest_path=mask_dest_path,
                                  is_mask=True)

    #gl.save_all_anno_images()

    gl.get_images_no_mask()


def train_unet(): #TODO update to use current versions
    train_path = r'D:\Clarifruit\cherry_stem\data\unet_data\train'
    test_path = r'D:\Clarifruit\cherry_stem\data\unet_data\test'
    test_aug_path = os.path.join(test_path, 'aug')

    target_size = (256, 256)
    modes_dict = {'grayscale': 1, 'rgb': 3}

    color_mode = 'rgb'

    x_folder_name = 'image'
    y_folder_name = 'label'

    x_prefix = 'image'
    y_prefix = 'label'

    weights_file_name = 'unet_cherry_stem.hdfs5'
    input_size = (*target_size, modes_dict[color_mode])

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    train_gen = clarifruit_train_generator(batch_size=2,
                                           train_path=train_path,
                                           image_folder=x_folder_name,
                                           mask_folder=y_folder_name,
                                           aug_dict=data_gen_args,
                                           image_color_mode=color_mode,
                                           mask_color_mode=color_mode,
                                           image_save_prefix=x_prefix,
                                           mask_save_prefix=y_prefix,
                                           save_to_dir=None,
                                           target_size=target_size,
                                           seed=1)

    model = unet(input_size=input_size,pretrained_weights=weights_file_name)
    model_checkpoint = ModelCheckpoint(weights_file_name, monitor='loss', verbose=1, save_best_only=True)
    #model.fit_generator(train_gen, steps_per_epoch=100, epochs=3, callbacks=[model_checkpoint])

    test_path_image = os.path.join(test_path,x_folder_name)
    pred_path = os.path.join(test_path,'pred')
    prediction(model, test_path_image, pred_path, target_size,as_gray=False)




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
    img_color='color' # keep color at the moment doesnt work with grayscale

    # fiz segmentation parameters
    scale = 100
    sigma = 0.5

    min_size = 100

    #mask_draw_params
    color=(255,0,255)
    alpha=1

    segmentation.segment(image_name, orig_path, mask_path, seg_path, seg_folder, seg_activation_folder,
             threshold, scale, sigma, min_size,color,alpha,boundaries_display_flag,save_flag,img_color)


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


    segmentation.segment_multi(orig_path, mask_path, seg_path, seg_folder, seg_activation_folder,difficult_list,
        threshold=threshold, scale=scale, sigma=sigma, min_size=min_size)



def visualize():
    image_name = '45665-81662.png.jpg'

    orig_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\image'

    img_path = os.path.join(orig_path,image_name)
    sv = seg_filter.SegmentFilter(img_path)
    sv.display_sigmentation_filter()

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def hsv():
    image_name = '45665-81662.png.jpg'

    orig_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\image'
    img_color='color'

    img_path = os.path.join(orig_path, image_name)
    img = cv2.imread(img_path, segmentation.COLOR_DICT[img_color])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #segmentation.visualize_rgb(img)
    #segmentation.visualize_hsv(img)
    #codes from http://www.workwithcolor.com/green-color-hue-range-01.htm
    #left = (178,236,93) # color Inchworm
    #right = (65,72,51) #Rifle Green
    #segmentation.color_thres(img, left, right)


def roman_segment():
    image_name = '45665-81662.png.jpg'

    orig_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\image'
    img_color='color'

    img_path = os.path.join(orig_path, image_name)
    img = cv2.imread(img_path, segmentation.COLOR_DICT[img_color])


def main():
    annotate()
    #train_unet()
    #activate_segmentation()
    #get_multi_segments()
    #visualize()
    #hsv()

if __name__ == "__main__":
    main()
