from work.unet import unet_model_functions

from keras.callbacks import ReduceLROnPlateau

from logger_settings import *

configure_logger()
logger = logging.getLogger("unet_main")


def load_from_files(src_path, updat_dict=None):
    """
    a method that can load a trained instance of ClarifruitUnet from a path in src_path variable
    the model parameters can be modified for further training as shown within
    :param src_path: the path where the trained model resides in
    :return: the ClarifruitUnet instance
    """
    params_dict = unet_model_functions.ClarifruitUnet.load_model(src_path)
    if updat_dict is not None:
        params_dict.update(updat_dict)
    logger.info(f"loading model from {src_path}")
    model = unet_model_functions.ClarifruitUnet(**params_dict)
    return model, params_dict


def main():
    train_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes'
    #train_path = r'D:\Clarifruit\unet\data\membrane\train'
    dest_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training'
    test_path = r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    #test_path =r'D:\Clarifruit\unet\data\membrane\test'
    src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-10-14_17-22-12'
    weights_name = 'unet_cherry_stem.steps_506-loss_0.12.hdf5'

    params_dict = dict(

        train_path=train_path,
        save_path=dest_path,
        x_folder_name='image',
        y_folder_name='label',
        weights_file_name='unet_cherry_stem.hdf5',

        data_gen_args=dict(rotation_range=180,
                           brightness_range=[0.2, 1.],
                           shear_range=10,
                           zoom_range=0.5,
                           horizontal_flip=True,
                           vertical_flip=True,
                           channel_shift_range=5,
                           fill_mode='nearest'),
        seed=78,

        optimizer='Adam',
        optimizer_params=dict(lr=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy'],

        target_size=(256, 256),
        color_mode='grayscale',

        batch_size=10,  # my GPU cant handel any more

        epochs=2,
        steps_per_epoch=30,
        validation_split=0.2,
        validation_steps=30,

        tensorboard_update_freq=1000,
        weights_update_freq=100,
        ontop_display_threshold=0.4)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=1, min_lr=0.000001,
                                  cooldown=1, verbose=1)


    # callbacks = [reduce_lr, early_stoping]
    # params_dict['callbacks'] = callbacks

    # model = train_model(train_path=train_path,
    #                     params_dict=params_dict,
    #                     dest_path=dest_path)


    # load model
    # params_dict = unet_model_functions.ClarifruitUnet.load_model(src_path, weights_name)
    # params_dict.update(params_dict1)
    #model = unet_model_functions.ClarifruitUnet(**params_dict)
    model =unet_model_functions.ClarifruitUnet.load_model(src_path,weights_name)
    keras_logs_path = model.set_model_for_train()
    model.fit_unet()

if __name__ == '__main__':
    main()
