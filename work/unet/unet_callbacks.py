import keras
from PIL import Image
import io
import numpy as np
from keras.callbacks import Callback
from tensorflow.compat.v1 import Summary,summary
from auxiliary.custom_image import CustomImage
from logger_settings import *

configure_logger()
logger = logging.getLogger("unet_callbacks")


class ImageHistory(Callback):
    def __init__(self, tensor_board_dir, data, last_step=0, draw_interval=100):

        super().__init__()
        self.last_step = last_step
        self.draw_interval = draw_interval

        self.writer = summary.FileWriter(tensor_board_dir)

        self.get_data(data)

    def get_data(self,data):
        self.data = data[0]

        for i in range(5):
            image_data = data[0][i]
            label_data = data[1][i]

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



    def make_image(self, tensor):
        """
        Convert an numpy representation image to Image protobuf.
        taken and updated from https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = tensor.shape
        if channel == 1:
            tensor = np.squeeze(tensor,axis=2)
        image = Image.fromarray(tensor)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return Summary.Image(height=height, width=width, colorspace=channel,
                                          encoded_image_string=image_string)



    def to_image(self,data):
        img = (255 * data).astype(np.uint8)
        return self.make_image(img)

    def saveToTensorBoard(self, image, tag, batch=None):

        image_summary = Summary.Value(tag=tag, image=image)
        summary_value = Summary(value=[image_summary])

        self.writer.add_summary(summary_value, global_step=batch)


    def on_batch_end(self, batch, logs=None):

        if logs is None:
            logs = {}

        if batch % self.draw_interval == 0:
            logger.info("updating tensorboard images,step:" + str(self.last_step))
            #self.writer.add_text('Text', 'text logged at step:' + str(self.last_step), self.last_step)

            #epoch = self.last_step * self.draw_interval

            y_pred = self.model.predict(self.data)
            for i in range(5):
                img = self.data[i].astype(np.uint8)
                raw_pred = y_pred[i]
                custom_img = CustomImage(img=img,mask=raw_pred,threshold=0.5)
                custom_img.get_ontop()

                proto_ontop= self.make_image(custom_img.ontop)
                tag = f'plot_{i}/ontop'
                self.saveToTensorBoard(image=proto_ontop,
                                       tag=tag,
                                       batch=self.last_step)

                proto_pred= self.to_image(raw_pred)
                tag = f'plot_{i}/pred'
                self.saveToTensorBoard(image=proto_pred,
                                       tag=tag,
                                       batch=self.last_step)
        self.last_step += 1
    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        # Arguments
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        self.writer.close()
