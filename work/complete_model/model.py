from work.unet.unet_model_functions import ClarifruitUnet
from work.auxiliary import data_functions
from work.stem_extraction.stem_extract import CustomImageExtractor
import os
from work.stem_classifier.classify import StemHistClassifier


class TrainedModel:
    """
    ---EXPERIMENTAL---
    A class for a full prediction pipeline from trained models
    """
    def __init__(self, unet_model_path, classifier_path,
                 augmentation_params=None):
        self.unet_model_path = unet_model_path
        self.classifier_path = classifier_path
        self.augmentation_params = augmentation_params

        self.unet_model = None
        self.cls_model = None

        self.threshold = None
        self.hist_type = None

        self.load_models()

    def load_models(self):

        self.unet_model = ClarifruitUnet.load_model(self.unet_model_path)
        for item_entry in os.scandir(self.classifier_path):
            if item_entry.name == 'extraction_params.json':
                params = data_functions.load_json(item_entry.path)
                self.threshold = float(params['threshold'])
                self.hist_type = params['hist_type']
            if item_entry.name.split('.')[-1] == 'pickle':
                self.cls_model = data_functions.load_pickle(item_entry.path)

    def predict(self, test_path):
        pred_list = []
        for img_entry, pred in self.unet_model.prediction_generator(test_path):
            curr_image = CustomImageExtractor(img_path=img_entry.path,
                                              threshold=self.threshold,
                                              mask=pred,
                                              create_save_dest_flag=False,
                                              **self.augmentation_params)
            if self.augmentation_params is not None:
                curr_image.binary_mask = curr_image.get_ontop_seg(
                    save_flag=False)

            hist = curr_image.get_hist_via_mask(hist_type=self.hist_type)
            hist = StemHistClassifier.return_hist(hist, self.hist_type).reshape(
                1, -1)
            pred = self.cls_model.predict(hist)[0]
            pred_list.append((img_entry, pred))
        return pred_list
