from PIL import Image
import io
import numpy as np
from keras.callbacks import Callback, TensorBoard
from tensorflow.compat.v1 import Summary
import warnings
from work.auxiliary.custom_image import CustomImage
from work.auxiliary import decorators
from work.auxiliary.logger_settings import *
import os
import re

logger = logging.getLogger("unet_callbacks")
logger_decorator = decorators.Logger_decorator(logger)


class CustomTensorboardCallback(TensorBoard):

    @logger_decorator.debug_dec
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch',
                 data=None,
                 threshold=0.5,
                 samples_seen=0):

        super().__init__(log_dir,
                         histogram_freq,
                         batch_size,
                         write_graph,
                         write_grads,
                         write_images,
                         embeddings_freq,
                         embeddings_layer_names,
                         embeddings_metadata,
                         embeddings_data,
                         update_freq)
        self.data = data
        self.threshold = threshold
        self.samples_seen = samples_seen

    @logger_decorator.debug_dec
    def set_model(self, model):
        super().set_model(model)
        if self.data is not None:
            self.get_data()

    @logger_decorator.debug_dec
    def get_data(self):
        """
        Create base images and labels for tensorboard visualization from given
        validation set also sets the data as an input tensor for showing
        prediction progress
        :return:
        """
        data_list = []
        for i, data in enumerate(self.data):
            image_data = data[0][5]
            label_data = data[1][5]
            data_list.append(image_data[np.newaxis, :])
            raw_image = (255 * image_data).astype(np.uint8)
            raw_mask = (255 * label_data).astype(np.uint8)

            proto_image = self.make_image(raw_image)
            image_tag = f'plot_{i}/image'
            self.saveToTensorBoard(image=proto_image,
                                   tag=image_tag)

            proto_ground_truth = self.make_image(raw_mask)
            image_tag = f'plot_{i}/ground_truth'
            self.saveToTensorBoard(image=proto_ground_truth,
                                   tag=image_tag)

        self.data = np.concatenate(data_list, axis=0)

    @logger_decorator.debug_dec
    def make_image(self, tensor):
        """
        Convert an numpy representation image to Image protobuf.
        taken and updated from https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = tensor.shape
        if channel == 1:
            image = Image.fromarray(tensor[:, :, 0])
        else:
            image = Image.fromarray(tensor)
        output = io.BytesIO()
        image.save(output, format='JPEG')
        image_string = output.getvalue()
        output.close()
        return Summary.Image(height=height, width=width, colorspace=channel,
                             encoded_image_string=image_string)

    @logger_decorator.debug_dec
    def saveToTensorBoard(self, image, tag, batch=None):
        """
        save a protobuf image with a given tag with the writter class
        :param image: a protobuf image
        :param tag: str, the tag of the image
        :param batch: the current number of steps
        :return:
        """

        image_summary = Summary.Value(tag=tag, image=image)
        summary_value = Summary(value=[image_summary])

        self.writer.add_summary(summary_value, global_step=batch)

    @logger_decorator.debug_dec
    def on_batch_end(self, batch, logs=None):
        """
        modify the Tensorboard on_batch_end to get predictions from self.data
        at the predefined update frequency

        """
        if self.update_freq != 'epoch':
            self.samples_seen += logs['size']
            samples_seen_since = self.samples_seen - \
                                 self.samples_seen_at_last_write
            if samples_seen_since >= self.update_freq:
                self._write_logs(logs, self.samples_seen)
                self.samples_seen_at_last_write = self.samples_seen
                if self.data is not None:
                    batch_size = self.data.shape[0]
                    y_pred = self.model.predict(self.data,
                                                batch_size=batch_size)
                    for i in range(batch_size):
                        img = (255 * self.data[i]).astype(np.uint8)
                        raw_pred = y_pred[i]

                        custom_img = CustomImage(img=img, raw_mask=raw_pred,
                                                 threshold=self.threshold,
                                                 create_save_dest_flag=False)

                        raw_pred_img = (255 * raw_pred).astype(np.uint8)

                        proto_pred_raw = self.make_image(raw_pred_img)
                        proto_pred_binary = self.make_image(
                            custom_img.binary_mask)
                        proto_ontop = self.make_image(custom_img.get_ontop())

                        raw_pred_tag = f'plot_{i}/raw_pred'
                        self.saveToTensorBoard(image=proto_pred_raw,
                                               tag=raw_pred_tag,
                                               batch=self.samples_seen)

                        binary_pred_tag = f'plot_{i}/binary_pred_thres_' \
                                          f'{self.threshold}'
                        self.saveToTensorBoard(image=proto_pred_binary,
                                               tag=binary_pred_tag,
                                               batch=self.samples_seen)

                        onrop_tag = f'plot_{i}/ontop_threshold_{self.threshold}'
                        self.saveToTensorBoard(image=proto_ontop,
                                               tag=onrop_tag,
                                               batch=self.samples_seen)


class CustomModelCheckpoint(Callback):
    @logger_decorator.debug_dec
    def __init__(self, filepath, monitor='loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 update_freq=1000, batch_size=10, samples_seen=0,
                 model_params_path=None,
                 session_n=1, period=1):

        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only

        self.batch_size = batch_size
        self.update_freq = update_freq
        self.samples_seen = samples_seen
        self.samples_seen_at_last_write = 0
        self.model_params_path = model_params_path
        self.sessions_n = session_n
        self.period = period
        self.epochs_since_last_save = 0

        if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.less
            self.best = np.Inf

    @logger_decorator.debug_dec
    def modify_params_file(self):
        dirname, basename = os.path.split(self.model_params_path)
        new_file_name = re.sub(r"(steps_)(\d+)(\.)",
                               f"\g<1>{self.samples_seen}\g<3>",
                               basename)
        new_model_params_path = os.path.join(dirname, new_file_name)
        os.rename(self.model_params_path, new_model_params_path)
        self.model_params_path = new_model_params_path

    @logger_decorator.debug_dec
    def on_batch_end(self, batch, logs=None):
        """
        modifiy the Tensorboard on_batch_end to get predictions from self.data
        at the predefined update frequency
        :param batch: batch number
        :param logs:
        :return:
        """
        self.samples_seen += logs['size']
        if self.update_freq != 'epoch':
            samples_seen_since = self.samples_seen - \
                                 self.samples_seen_at_last_write
            if samples_seen_since >= self.update_freq:

                self.modify_params_file()

                self.samples_seen_at_last_write = self.samples_seen
                filepath = self.filepath.format(steps=self.samples_seen,
                                                sess=self.sessions_n,
                                                **logs)
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn(
                            'Can save best model only with %s available, '
                            'skipping.' % (self.monitor), RuntimeWarning)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nstep %05d: %s improved from %0.5f '
                                      'to %0.5f, saving model to %s' % (
                                          self.samples_seen, self.monitor,
                                          self.best,
                                          current, filepath))

                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath,
                                                        overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('\nsteps %05d: %s did not improve from'
                                      ' %0.5f' % (
                                        self.samples_seen, self.monitor,
                                        self.best))
                else:
                    if self.verbose > 0:
                        print('\nstep %05d: saving model to %s' % (
                            self.samples_seen, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)

    @logger_decorator.debug_dec
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.update_freq == 'epoch':
            self.modify_params_file()
            if self.epochs_since_last_save >= self.period:
                self.epochs_since_last_save = 0
                filepath = self.filepath.format(steps=self.samples_seen,
                                                sess=self.sessions_n,
                                                **logs)
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn(
                            'Can save best model only with %s available, '
                            'skipping.' % (self.monitor), RuntimeWarning)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print(
                                    '\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                    ' saving model to %s'
                                    % (epoch + 1, self.monitor, self.best,
                                       current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath,
                                                        overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print(
                                    '\nEpoch %05d: %s did not improve from %0.5f' %
                                    (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (
                            epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)
