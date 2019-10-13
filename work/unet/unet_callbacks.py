import keras
from PIL import Image
import io
import numpy as np
from keras.callbacks import TensorBoard
from tensorflow.compat.v1 import Summary, summary
from auxiliary.custom_image import CustomImage
from logger_settings import *

configure_logger()
logger = logging.getLogger("unet_callbacks")


class ImageHistory(TensorBoard):
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
                 data=None):

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


    def set_model(self, model):
        super().set_model(model)
        if self.data is not None:
            self.get_data()


    def get_data(self):
        data_list = []
        for i,data in enumerate(self.data):
            image_data = data[0][5]
            label_data = data[1][5]
            data_list.append(image_data[np.newaxis,:])
            raw_image = image_data.astype(np.uint8)
            raw_mask = label_data.astype(np.uint8)

            proto_image = self.make_image(raw_image)
            image_tag = f'plot_{i}/image'
            self.saveToTensorBoard(image=proto_image,
                                   tag=image_tag)

            proto_ground_truth = self.make_image(raw_mask)
            image_tag = f'plot_{i}/ground_truth'
            self.saveToTensorBoard(image=proto_ground_truth,
                                   tag=image_tag)

        self.data = np.concatenate(data_list,axis=0)
        #self.data = data_list

    def make_image(self, tensor):
        """
        Convert an numpy representation image to Image protobuf.
        taken and updated from https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = tensor.shape
        if channel == 1:
            #image = Image.frombytes('L', (width, height), tensor.tobytes())
            image = Image.fromarray(tensor[:,:,0])
        else:
            image = Image.fromarray(tensor)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return Summary.Image(height=height, width=width, colorspace=channel,
                             encoded_image_string=image_string)

    def to_image(self, data):
        img = (255 * data).astype(np.uint8)
        return self.make_image(img)

    def saveToTensorBoard(self, image, tag, batch=None):

        image_summary = Summary.Value(tag=tag, image=image)
        summary_value = Summary(value=[image_summary])

        self.writer.add_summary(summary_value, global_step=batch)

    def on_batch_end(self, batch, logs=None):

        if self.update_freq != 'epoch':
            self.samples_seen += logs['size']
            samples_seen_since = self.samples_seen - self.samples_seen_at_last_write
            if samples_seen_since >= self.update_freq:
                self._write_logs(logs, self.samples_seen)
                self.samples_seen_at_last_write = self.samples_seen
                if self.data is not None:
                    batch_size = self.data.shape[0]
                    #logger.info("updating tensorboard images,step:" + str(self.samples_seen))
                    y_pred = self.model.predict(self.data,batch_size=batch_size)
                    for i in range(batch_size):
                        #raw_pred = self.model.predict(self.data[i])[0]
                        img = self.data[i].astype(np.uint8)
                        raw_pred = y_pred[i]
                        custom_img = CustomImage(img=img, mask=raw_pred, threshold=0.5)
                        custom_img.get_ontop()
                        proto_ontop = self.make_image(custom_img.ontop)

                        tag = f'plot_{i}/ontop'
                        self.saveToTensorBoard(image=proto_ontop,
                                               tag=tag,
                                               batch=self.samples_seen)

                        raw_pred_img = (255 * raw_pred).astype(np.uint8)
                        proto_pred = self.make_image(raw_pred_img)
                        tag = f'plot_{i}/pred'
                        self.saveToTensorBoard(image=proto_pred,
                                               tag=tag,
                                               batch=self.samples_seen)

