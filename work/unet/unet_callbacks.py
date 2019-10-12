import keras
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from tensorflow.compat.v1 import Summary
from auxiliary.custom_image import CustomImage

class ImageHistory(Callback):
    def __init__(self, tensor_board_dir, data, last_step=0, draw_interval=100):
        self.get_data(data)

        self.last_step = last_step
        self.draw_interval = draw_interval
        self.tensor_board_dir = tensor_board_dir

    def get_data(self,data):
        data_list = []
        for item in data:
            image_data = item[0]
            label_data = item[1]

            image = self.to_image(image_data[0])
            ground_truth = self.to_image(label_data[0])
            data_list.append((image_data,image,ground_truth))
        self.data =data_list

    def make_image(self, npyfile):
        """
        Convert an numpy representation image to Image protobuf.
        taken and updated from https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = npyfile.shape
        image = Image.frombytes('L', (width, height), npyfile.tobytes())
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return Summary.Image(height=height, width=width, colorspace=channel,
                                          encoded_image_string=image_string)

    # def saveToTensorBoard(self, image,tag, epoch):
    #
    #     image = self.make_image(image)
    #     summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, image=image)])
    #     writer = tf.compat.v1.summary.FileWriter(self.tensor_board_dir)
    #     writer.add_summary(summary, epoch)
    #     writer.close()

    def to_image(self,data):
        image = (((data - data.min()) * 255) / (data.max() - data.min())).astype(np.uint8)
        return self.make_image(image)

    def saveToTensorBoard(self, image,ground_truth,pred, epoch):
        image_summary = Summary.Value(tag='image', image=image)
        ground_truth_summary = Summary.Value(tag='image', image=ground_truth)
        pred_summary = Summary.Value(tag='image', image=pred)

        summary = Summary(value=[image_summary,ground_truth_summary,pred_summary])
        writer = summary.FileWriter(self.tensor_board_dir)
        writer.add_summary(summary, epoch)
        writer.close()

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.draw_interval == 0:
            epoch = self.last_step * self.draw_interval
            self.last_step += 1
            for image_data,image,ground_truth in self.data:

                y_pred = self.model.predict(image_data)
                pred= self.to_image(y_pred[0])

                self.saveToTensorBoard(image=image,
                                       ground_truth=ground_truth,
                                       pred=pred,
                                       epoch=epoch)

                break