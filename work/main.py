from work.annotation import fruits_anno
from work.unet.data_functions import *
#from work.unet.model import *
from work.segmentation.segmentation import *

from tqdm import tqdm

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

    gl.save_all_anno_images()


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
    threshold = 10  # for the segmenation folder

    # fiz segmentation parameters
    scale = 200
    sigma = 0.5

    min_size = 100

    #mask_draw_params
    color=(255,0,255)
    alpha=1

    segment(image_name, orig_path, mask_path, seg_path, seg_folder, seg_activation_folder,
             threshold, scale, sigma, min_size,color,alpha,boundaries_display_flag,save_flag)


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


    threshold = 10  # for the segmenation folder

    scale = 200
    sigma = 0.5

    min_size = 70


    segment_multi(orig_path, mask_path, seg_path, seg_folder, seg_activation_folder,difficult_list,
        threshold=threshold, scale=scale, sigma=sigma, min_size=min_size)

def main():
    #annotate()
    #train_unet()
    activate_segmentation()
    #get_multi_segments()

if __name__ == "__main__":
    main()
