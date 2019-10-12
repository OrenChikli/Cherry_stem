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
        self.get_data(data)

        self.last_step = last_step
        self.draw_interval = draw_interval
        self.tensor_board_dir = tensor_board_dir

    def get_data(self,data):
        data_list = []

        for i in range(5):
            image_data = data[0][i]
            label_data = data[1][i]

            raw_image = image_data.astype(np.uint8)
            raw_mask = label_data.astype(np.uint8)

            proto_image = self.make_image(raw_image)
            proto_ground_truth = self.make_image(raw_mask)

            data_list.append((image_data,proto_image,proto_ground_truth))

        self.data = data_list

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

    def saveToTensorBoard(self,i, image,ground_truth,pred, epoch):

        image_summary = Summary.Value(tag=f'plot_{i}/image', image=image)
        ground_truth_summary = Summary.Value(tag=f'plot_{i}/ground_truth', image=ground_truth)
        pred_summary = Summary.Value(tag=f'plot_{i}/pred', image=pred)

        summary_value = Summary(value=[image_summary,ground_truth_summary,pred_summary])
        with summary.FileWriter(self.tensor_board_dir) as writer:
            writer.add_summary(summary_value, epoch)


    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.draw_interval == 0:
            epoch = self.last_step * self.draw_interval
            self.last_step += 1
            for i,(image_data,proto_image,proto_ground_truth) in enumerate(self.data):

                y_pred = self.model.predict(image_data[np.newaxis,:])
                pred= self.to_image(y_pred[0])

                self.saveToTensorBoard(image=proto_image,
                                       ground_truth=proto_ground_truth,
                                       pred=pred,
                                       epoch=epoch,
                                       i=i)

