
#from .clarifruit_unet.data_functions import *
#from .clarifruit_unet.unet_model import *
from work.unet.clarifruit_unet.keras_functions import *
#from datetime import datetime


src_path = r'D:\Clarifruit\cherry_stem\data\unet_data\no_thick_lines\Data split\test_split_0.3'

train_folder = 'train'
test_folder = 'test'

x_folder_name = 'image'
y_folder_name = 'label'



train_path = os.path.join(src_path ,train_folder)
test_path = os.path.join(src_path ,test_folder)



model_save_dest_path = r'D:\Clarifruit\cherry_stem\data\unet_data\model data'

modes_dict = {'grayscale': 1, 'rgb': 3}  # translate for image dimentions

target_size = (256 ,256)
color_mode = 'rgb'

weights_file_name = 'unet_cherry_stem.hdf5'


data_gen_args = dict(rescale=1./255,
                     rotation_range=0.5,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

fit_params = dict(batch_size = 10,
                  epochs = 1,
                  steps_per_epoch = 10,
                  validation_steps = 10)



params_dict = dict(src_path = train_path,
                   dest_path = model_save_dest_path,
                   
                   x_folder_name = x_folder_name,
                   y_folder_name = y_folder_name,
                   
                   
                   target_size = target_size,
                   color_mode = color_mode,
                   input_size = (*target_size, modes_dict[color_mode]),
                   
                   weights_file_name = weights_file_name,
                   
                   data_gen_params = data_gen_args,
                   model_fit_params = fit_params )

# early_stoping = EarlyStopping(monitor='val_loss',verbose=1, patience=3)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=2, min_lr=0.000001,
                              cooldown=1 ,verbose=1)
# callbacks = [early_stoping, model_checkpoint,reduce_lr]
callbacks = [reduce_lr]



model ,curr_time = clarifruit_train(params_dict ,callbacks)




# for the prediction part
pred_path = create_path(src_path ,f'test_pred_{curr_time}')
image_train_path = os.path.join(train_path,x_folder_name)


prediction(model, image_train_path, pred_path, target_size ,threshold=0.5 ,as_gray=False)







