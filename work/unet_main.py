import os
from work.unet import unet_model_functions
from keras.callbacks import ReduceLROnPlateau
from auxiliary import data_functions
from logger_settings import configure_logger

REAL_PATH = os.path.abspath('..')
LOG_PATH = os.path.join(REAL_PATH,'logs')

log_path = data_functions.create_path(LOG_PATH,'unet_logs')
logger = configure_logger(name="cherry_stem",
                          console_level='INFO',
                          file_level='INFO',
                          out_path=log_path)


def main():
    train_path = os.path.join(REAL_PATH, r'data\raw_data\with_maskes')
    dest_path = os.path.join(REAL_PATH, r'data\unet_data\training')
    test_path = os.path.join(REAL_PATH, r'data\raw_data\images_orig')

    src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-10-17_14-38-00'
    steps = None

    params_dict = dict(

        train_path=train_path,
        save_path=dest_path,
        x_folder_name='image',
        y_folder_name='label',

        save_name='cherry',

        pretrained_weights=r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-10-16_16-48-42\cherry_ckpt.train_sess_01.steps_7132.loss_0.0270.hdf5',
        checkpoint=None,
        data_gen_args=dict(rotation_range=180,
                           brightness_range=[0.2, 1.],
                           shear_range=5,
                           zoom_range=0.5,
                           horizontal_flip=True,
                           vertical_flip=True,
                           fill_mode='nearest'),
        seed=78,

        optimizer='Adam',
        optimizer_params=dict(lr=1e-6,
                              amsgrad=True),

        loss='binary_crossentropy',
        metrics=[],

        target_size=[256, 256],
        color_mode='grayscale',

        batch_size=10,  # my GPU cant handel any more

        epochs=10,
        steps_per_epoch=2000,
        validation_split=0.2,
        validation_steps=200,

        tensorboard_update_freq=1000,
        weights_update_freq='epoch',
        save_weights_only=False,
        ontop_display_threshold=0.4,
        callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                                     verbose=1, mode='auto', min_delta=0.0001,
                                     cooldown=0, min_lr=1e-6)])

    update_dict=None
    update_dict = dict(callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.8, patience=2,
                                                    verbose=1, mode='auto', min_delta=0.0001,
                                                    cooldown=0, min_lr=1e-6)])


    model = unet_model_functions.ClarifruitUnet(**params_dict)

    #model = unet_model_functions.ClarifruitUnet.load_model(src_path,update_dict,steps)

    model.set_model_for_train()
    model.fit_unet()


if __name__ == '__main__':
    main()
