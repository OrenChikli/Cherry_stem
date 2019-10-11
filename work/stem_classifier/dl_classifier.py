from keras.models import *
from keras.layers import *
from keras.optimizers import *
#from keras import backend as keras
import logging
import tensorflow.compat.v1.logging as tf_logging # to stop tensorflow from displaying depracetion messages
tf_logging.set_verbosity(tf_logging.ERROR)

logger = logging.getLogger(__name__)


def class_model(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=('accuracy'),
         pretrained_weights=None, input_size=(256, 256, 1)):
    """
    an impelemntation of the unet model, taken from https://github.com/zhixuhao/unet
    :param optimizer:keras optimizer to use in the model
    :param loss: keras loss function
    :param metrics: metrics list for
    :param pretrained_weights: path to possible pretrained weights which can be loaded into the model
    :param input_size: the dimensions of the input images, defualt is (256,256,1) images
    :return:
    """
    logger.debug(f"<- unet model with input_size={input_size} andpretraind_weights={pretrained_weights} ")
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    drop5 = Dropout(0.5)(conv5)
    dense1 = Dense(128, activation='relu',kernel_initializer='he_normal')(drop5)
    flat = Flatten()(dense1)
    dense2 = Dense(4,activation='softmax',kernel_initializer='he_normal')(flat)

    model = Model(inputs=inputs, outputs=dense2)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model