from logger_settings import *

from work.unet import unet_model_functions

configure_logger()
logger = logging.getLogger("unet_main")



def main():
    train_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes'
    dest_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training'

    test_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'

    src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-10-14_23-59-57'
    steps = None

    params_dict = dict(

        train_path=train_path,
        save_path=dest_path,
        x_folder_name='image',
        y_folder_name='label',
        weights_file_name='unet_cherry_stem',

        data_gen_args=dict(rotation_range=180,
                           brightness_range=[0.2, 1.],
                           shear_range=5,
                           zoom_range=0.5,
                           horizontal_flip=True,
                           vertical_flip=True,
                           fill_mode='nearest'),
        seed=78,

        optimizer='Adam',
        optimizer_params=dict(lr=1e-4,
                              decay=1e-5),

        loss='binary_crossentropy',
        metrics=['accuracy'],

        target_size=(256, 256),
        color_mode='grayscale',

        batch_size=10,  # my GPU cant handel any more

        epochs=10,
        steps_per_epoch=4000,
        validation_split=0.2,
        validation_steps=400,

        tensorboard_update_freq=1000,
        weights_update_freq=5000,
        ontop_display_threshold=0.4)


    update_dict = dict(epochs=2,
                       steps_per_epoch=50,
                       validation_split=0.2,
                       validation_steps=50,

                       tensorboard_update_freq=1000,
                       weights_update_freq=100,
                       ontop_display_threshold=0.4)

    model = unet_model_functions.ClarifruitUnet(**params_dict)

    # model = unet_model_functions.ClarifruitUnet.load_model(src_path,steps)
    # model.update_model(**update_dict)

    model.set_model_for_train()
    model.fit_unet()


if __name__ == '__main__':
    main()
