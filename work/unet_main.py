from work.unet import unet_model_functions
from auxiliary import data_functions
from keras.callbacks import ReduceLROnPlateau


from logger_settings import *
configure_logger()
logger = logging.getLogger("unet_main")



def get_data_via_with_mask():
    src_path = r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes'
    data_path = r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines'
    src_folder = 'with_maskes'
    x_folder = 'image'
    y_folder = 'label'

    data_functions.get_from(src_path, data_path, src_folder, x_folder, y_folder)


def main():
    params_dict = dict(

        train_path=r'D:\Clarifruit\cherry_stem\data\raw_data\with_maskes',
        test_path=r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig',
        x_folder_name='image',
        y_folder_name='label',
        dest_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training',
        weights_file_name='unet_cherry_stem.hdf5',

        data_gen_args=dict(rescale=1. / 255,
                           rotation_range=180,
                           brightness_range=[0.2, 1.],
                           width_shift_range=0.25,
                           height_shift_range=0.25,
                           shear_range=0.2,
                           zoom_range=[0.5, 1.0],
                           horizontal_flip=True,
                           vertical_flip=True,
                           fill_mode='nearest'),

        optimizer='Adam',
        optimizer_params=dict(lr=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],
        pretrained_weights=None,

        target_size=(256, 256),
        color_mode='grayscale',
        mask_color_mode='grayscale',
        batch_size=10,
        epochs=2,
        steps_per_epoch=10,
        valdiation_split=0.2,
        validation_steps=10)


    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=2, min_lr=0.000001,
                                  cooldown=1, verbose=1)

    callbacks = [reduce_lr]
    params_dict['callbacks'] = callbacks
    logger.info("created training instance")
    model = unet_model_functions.ClarifruitUnet(**params_dict)
    logger.info("train start")
    model.train_model(params_dict=params_dict,saveflag=True)
    logger.info("prediction start")
    model.prediction()


def load_from_files():
    src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-27_16-15-36'
    params_dict = unet_model_functions.ClarifruitUnet.load_model(src_path)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=2, min_lr=0.000001,
                                  cooldown=1, verbose=1)

    train_params = dict(

        data_gen_args=dict(rescale=1. / 255,
                           rotation_range=180,
                           brightness_range=[0.2, 1.],
                           width_shift_range=0.25,
                           height_shift_range=0.25,
                           shear_range=0.2,
                           zoom_range=[0.5,1.0],
                           horizontal_flip=True,
                           vertical_flip=True,
                           fill_mode='nearest'),

        callbacks = [reduce_lr],

        batch_size=10,
        epochs=10,
        steps_per_epoch=3000,
        valdiation_split=0.2,
        validation_steps=3000)

    #params_dict.update(train_params)
    logger.info(f"loading model from {src_path}")
    model = unet_model_functions.ClarifruitUnet(**params_dict)

    #model.train_model(params_dict,saveflag=True)
    model.prediction()

def put_on():
    img_path=r'D:\Clarifruit\cherry_stem\data\raw_data\images_orig'
    mask_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-21_18-19-20\raw_pred'
    dest_path=r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-21_18-19-20\with_maskes'
    data_functions.get_with_maskes(img_path,mask_path,dest_path)

if __name__ == '__main__':
     #main()
     load_from_files()
     #put_on()
