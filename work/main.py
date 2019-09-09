#from work.annotation import fruits_anno
#from work.unet.data_functions import *
#from work.unet.model import *
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
                                           flag_multi_class=False,
                                           num_class=2,
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

    as_gray = True
    display_flag=True
    threshold = 1

    img_path = os.path.join(orig_path,image_name)
    mask_imgh_path = os.path.join(mask_path,image_name)
    # image = io.imread(img_path, as_gray=as_gray)
    # mask = io.imread(mask_imgh_path, as_gray=True)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_imgh_path, cv2.IMREAD_GRAYSCALE)
    mask_binary = np.where(mask==255,True,False)


    #weighted_img = mask_color_img(img, mask_binary, color=(0, 0, 255), alpha=0.5)
    #cv2.imshow("weighted", weighted_img)
    #cv2.waitKey(0)
    sg=Segmentation(image=img,ground_truth=mask_binary)
    sg.apply(display_flag=display_flag)
    seg_path='D:\Clarifruit\cherry_stem\data\segmentation'
    #curr_seg_path = os.path.join(seg_path,image_name)
    #if not os.path.exists(curr_seg_path):
        #os.mkdir(curr_seg_path)
    #sg.save_segments(curr_seg_path)
    #sg.filter_segments()
    sg.filter_segments(sg.filter_segments_by_prediction,threshold=threshold)

    #cv2.imshow("mask", mask)
    #cv2.waitKey(0)
    #cv2.imshow("Segmented", res_mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()






def main():
    #annotate()
    #train_unet()
    segment()


if __name__ == "__main__":
    main()
