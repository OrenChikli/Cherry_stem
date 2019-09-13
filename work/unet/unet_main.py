import os
from work.unet.clarifruit_unet import unet_model, keras_functions


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

    train_gen = keras_functions.clarifruit_train_generator(batch_size=2,
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

    model = unet_model.unet(input_size=input_size, pretrained_weights=weights_file_name)
    model_checkpoint = unet_model.ModelCheckpoint(weights_file_name, monitor='loss', verbose=1, save_best_only=True)
    #model.fit_generator(train_gen, steps_per_epoch=100, epochs=3, callbacks=[model_checkpoint])

    test_path_image = os.path.join(test_path,x_folder_name)
    pred_path = os.path.join(test_path,'pred')
    keras_functions.prediction(model, test_path_image, pred_path, target_size, as_gray=False)

def unet_main():
    pass

if __name__ == '__main__':
    pass