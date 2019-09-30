class ClarifruitModel:
    def __init__(self,unet_model_path,extraction_params,classification_params):
        self.unet_model_path = unet_model_path
        self.extraction_params=extraction_params
        self.classification_params = classification_params
