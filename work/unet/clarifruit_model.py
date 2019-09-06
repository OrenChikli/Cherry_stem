from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator

from .model import *

MODES_DICT = {'grayscale': 1, 'rgb': 3}

def adjustData(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3])) \
            if flag_multi_class else np.reshape(new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


def custom_generator(batch_size, train_path, folder, aug_dict, save_prefix,
                     color_mode="grayscale",
                     save_to_dir=None,
                     target_size=(256, 256),
                     seed=1):
    datagen = ImageDataGenerator(**aug_dict)
    gen = datagen.flow_from_directory(
        train_path,
        classes=[folder],
        class_mode=None,
        color_mode=color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=save_prefix,
        seed=seed)

    return gen


def clarifruit_train_generator(batch_size, train_path, image_folder, mask_folder, aug_dict,
                               image_color_mode="grayscale", mask_color_mode="grayscale",
                               image_save_prefix="image",
                               mask_save_prefix="mask",
                               flag_multi_class=False,
                               num_class=2,
                               save_to_dir=None,
                               target_size=(256, 256),
                               seed=1):
    """
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    """

    image_generator = custom_generator(batch_size=batch_size,
                                       train_path=train_path,
                                       folder=image_folder,
                                       aug_dict=aug_dict,
                                       save_prefix=image_save_prefix,
                                       color_mode=image_color_mode,
                                       save_to_dir=save_to_dir,
                                       target_size=target_size,
                                       seed=seed)

    mask_generator = custom_generator(batch_size=batch_size,
                                      train_path=train_path,
                                      folder=mask_folder,
                                      aug_dict=aug_dict,
                                      save_prefix=mask_save_prefix,
                                      color_mode=mask_color_mode,
                                      save_to_dir=save_to_dir,
                                      target_size=target_size,
                                      seed=seed)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


def clarifruit_test_generator(batch_size, test_path, folder, aug_dict, save_prefix,
                              color_mode="grayscale",
                              save_to_dir=None,
                              target_size=(256, 256),
                              seed=1):
    image_generator = custom_generator(batch_size=batch_size,
                                       train_path=test_path,
                                       folder=folder,
                                       aug_dict=aug_dict,
                                       save_prefix=save_prefix,
                                       color_mode=color_mode,
                                       save_to_dir=save_to_dir,
                                       target_size=target_size,
                                       seed=seed)

    for img in image_generator:
        if np.max(img) > 1:
            img = img / 255
        yield img


def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)


def main():
    train_path = 'data/membrane/train'
    test_path = 'data/membrane/test'
    target_size = (256, 256)
    color_mode = 'grayscale'

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    train_gen = clarifruit_train_generator(batch_size=2,
                                           train_path=train_path,
                                           image_folder='image',
                                           mask_folder='label',
                                           aug_dict=data_gen_args,
                                           image_color_mode=color_mode,
                                           mask_color_mode=color_mode,
                                           image_save_prefix="image",
                                           mask_save_prefix="mask",
                                           flag_multi_class=False,
                                           num_class=2,
                                           save_to_dir=None,
                                           target_size=target_size,
                                           seed=1)

    test_gen = clarifruit_train_generator(batch_size=2,
                                          train_path=test_path,
                                          image_folder='image',
                                          mask_folder='label',
                                          aug_dict=data_gen_args,
                                          image_color_mode=color_mode,
                                          mask_color_mode=color_mode,
                                          image_save_prefix="image",
                                          mask_save_prefix="mask",
                                          flag_multi_class=False,
                                          num_class=2,
                                          save_to_dir=None,
                                          target_size=target_size,
                                          seed=1)

    input_size = (*target_size, MODES_DICT[color_mode])
    model = unet(input_size=input_size)
    model_checkpoint = ModelCheckpoint('unet_cherry_stem.hdf5', monitor='loss', verbose=1, save_best_only=True)

    model.fit_generator(train_gen, steps_per_epoch=100, epochs=3, callbacks=[model_checkpoint])

    model.evaluate_generator(test_gen, steps=30, callbacks=[model_checkpoint], verbose=1)


    test = clarifruit_test_generator(batch_size=1,
                                     test_path=test_path,
                                     folder='image',
                                     aug_dict=data_gen_args,
                                     save_prefix='image',
                                     color_mode=color_mode,
                                     save_to_dir='data/test/aug',
                                     target_size=(256, 256),
                                     seed=1)

    results = model.predict_generator(generator=test, steps=30, verbose=1)
    saveResult(test_path, results)



if __name__ == '__main__':
    main()
