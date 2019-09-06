from work.annotation import fruits_anno
from work.unet.data_functions import *

def annotate():
    ano_path = r'D:\Clarifruit\cherry_stem\data\Cherry_cherry_stem_result.json'
    src_images_path = r'D:/Clarifruit/cherry_stem/data/images_orig/'
    dest_path = r'D:/Clarifruit/cherry_stem/data/masks/'
    csv_path = r'D:\Clarifruit\cherry_stem\data\cherry_stem.csv'


    gl = fruits_anno.GoogleLabels(anno_path=ano_path,
                                  src_images_path=src_images_path,
                                  csv_path=csv_path,
                                  dest_path=dest_path,
                                  is_mask=True)

    gl.save_all_anno_images()


def train_unet():
    train_path = r'D:\Clarifruit\cherry_stem\data\unet_data\train'
    test_path = r'D:\Clarifruit\cherry_stem\data\unet_data\test'
    test_aug_path = os.path.join(test_path, 'aug')

    target_size = (256, 256)
    modes_dict = {'grayscale': 1, 'rgb': 3}
    color_mode = 'grayscale'

    x_folder_name = 'image'
    y_folder_name = 'label'

    x_prefix = 'image'
    y_prefix = 'label'

    weights_file_name = 'unet_cherry_stem.hdfs5'

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

    test_gen = clarifruit_train_generator(batch_size=2,
                                          train_path=test_path,
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

    input_size = (*target_size, modes_dict[color_mode])
    model = unet(input_size=input_size)
    model_checkpoint = ModelCheckpoint(weights_file_name, monitor='loss', verbose=1, save_best_only=True)

    model.fit_generator(train_gen, steps_per_epoch=100, epochs=3, callbacks=[model_checkpoint])

    model.evaluate_generator(test_gen, steps=30, callbacks=[model_checkpoint], verbose=1)

    test = clarifruit_test_generator(batch_size=1,
                                     test_path=test_path,
                                     folder=x_folder_name,
                                     aug_dict=data_gen_args,
                                     save_prefix=x_prefix,
                                     color_mode=color_mode,
                                     save_to_dir=test_aug_path,
                                     target_size=input_size,
                                     seed=1)

    results = model.predict_generator(generator=test, steps=30, verbose=1)
    saveResult(test_path, results)

def main():
    train_unet()


if __name__ == "__main__":
    main()
