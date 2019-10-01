from work.unet.unet_model_functions import ClarifruitUnet
from auxiliary import data_functions
from auxiliary.custom_image import CustomImage
import os
from work.stem_classifier.classify import StemHistClassifier

class TrainedModel:
    def __init__(self,unet_model_path,classifier_path):
        self.unet_model_path = unet_model_path
        self.classifier_path = classifier_path

        self.unet_model = None
        self.cls_mosel=None

        self.threshold=None
        self.hist_type=None

        self.load_models()

    def load_models(self):

        params_dict = ClarifruitUnet.load_model(self.unet_model_path)
        self.unet_model = ClarifruitUnet(**params_dict)
        for item_entry in os.scandir(self.classifier_path):
            if item_entry.name == 'extraction_params.json':
                params = data_functions.load_json(item_entry.path)
                self.threshold = float(params['threshold'])
                self.hist_type = params['hist_type']
            if item_entry.name.split('.')[-1] == 'pickle':
                self.cls_model = data_functions.load_pickle(item_entry.path)

    def predict(self,test_path):
        pred_list = []
        for img_entry, pred in self.unet_model.prediction_generator(test_path):
            curr_image = CustomImage(img_path=img_entry.path, threshold=self.threshold, mask=pred)
            hist = curr_image.get_hist_via_mask(hist_type=self.hist_type)
            hist = StemHistClassifier.return_hist(hist, self.hist_type).reshape(1, -1)
            pred = self.cls_model.predict(hist)[0]
            pred_list.append((img_entry,pred))
        return pred_list

