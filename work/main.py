from work.annotation import fruits_anno
from work.unet.data_functions import *
from work.unet.model import *
from work.unet.segmentation import *
import cv2

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


def segment():
    image_name = '74714-32897.png.jpg'

    orig_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\image'
    mask_path = r'D:\Clarifruit\cherry_stem\data\unet_data\orig\label'
    seg_path = r'D:\Clarifruit\cherry_stem\data\segmentation'

    seg_folder = 'segments'
    seg_activation_folder = 'activation'

    # segmentaion paths
    curr_seg_path = create_path(seg_path, image_name)

    curr_segments_path = create_path(curr_seg_path, seg_folder)
    curr_activation_path = create_path(curr_seg_path,seg_activation_folder)


    boundaries_display_flag=False
    threshold = 1 #for the segmenation folder

    #load the src image and mask image
    img_path = os.path.join(orig_path,image_name)
    mask_imgh_path = os.path.join(mask_path,image_name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_imgh_path, cv2.IMREAD_GRAYSCALE)
    mask_binary = np.where(mask==255,True,False) #create binary version of the mask image


    #segmentation enhancment
    sg=Segmentation(image=img,ground_truth=mask_binary)
    sg.apply_segmentation(display_flag=boundaries_display_flag)
    sg.save_segments(curr_segments_path)
    seg_activation = sg.filter_segments(threshold=threshold)
    curr_activation_full = os.path.join(curr_activation_path,f'thres_{threshold}.jpg')
    cv2.imwrite(curr_activation_full,binary_to_grayscale(seg_activation))

    # show on source_image
    color=(0,0,255)
    alpha=0.8
    binary_seg_activation = np.where
    weighted = mask_color_img(img, seg_activation, color, alpha)
    plt.imshow(cv2.cvtColor(weighted, cv2.COLOR_BGR2RGB))
    plt.show()






def main():
    #annotate()
    #train_unet()
    segment()


if __name__ == "__main__":
    main()
