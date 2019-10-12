import keras
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from tensorflow.compat.v1 import Summary,summary
from auxiliary.custom_image import CustomImage

class ImageHistory(Callback):
    def __init__(self, tensor_board_dir, data, last_step=0, draw_interval=100):

        super().__init__()
        self.last_step = last_step
        self.draw_interval = draw_interval
        self.writer = tf.summary.FileWriter(tensor_board_dir)

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

    def saveToTensorBoard(self,image,tag, epoch=None):

        image_summary = Summary.Value(tag=tag, image=image)
        summary_value = Summary(value=[image_summary])

        self.writer.add_summary(summary_value, global_step=epoch)


    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.draw_interval == 0:
            epoch = self.last_step * self.draw_interval
            self.last_step += 1
            y_pred = self.model.predict(self.data)
            for i in range(5):
                raw_pred = y_pred[i]
                binary
                proto_pred= self.to_image(curr_pred)
                tag = f'plot_{i}/pred'
                self.saveToTensorBoard(image=proto_pred,
                                       tag=tag,
                                       epoch=epoch)

