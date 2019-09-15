from work.annotation.clarifruit_annotation import fruits_anno
from work.unet.clarifruit_unet.data_functions import *
#from work.unet.model import *
#from work.segmentation.segmentation import *
import logging



def annotate():
    raw_data_path = r'D:\Clarifruit\cherry_stem\data\raw_data'

    ano_path = os.path.join(raw_data_path, 'Cherry_cherry_stem_result.json')
    src_images_path = os.path.join(raw_data_path, 'images_orig')
    csv_path = os.path.join(raw_data_path, 'cherry_stem.csv')

    model_out_path = r'D:\Clarifruit\cherry_stem\data\annotation_output'

    mask_dest_path = os.path.join(model_out_path, 'masks')
    images_dest_path = os.path.join(model_out_path, 'train')

    gl = fruits_anno.GoogleLabels(anno_path=ano_path,
                                  src_images_path=src_images_path,
                                  csv_path=csv_path,
                                  dest_images_path=images_dest_path,
                                  mask_dest_path=mask_dest_path,
                                  is_mask=True)

    #gl.save_all_anno_images()

    gl.get_images_no_mask()









def main():
    #annotate()
    #train_unet()
    #activate_segmentation()
    #get_multi_segments()
    #visualize()
    #hsv()
    #print('hello')
    #roman_segment()
    pass

if __name__ == "__main__":
    # Gets or creates a logger
    logger = logging.getLogger(__name__)

    # set log level
    logger.setLevel(logging.WARNING)

    # define file handler and set formatter
    file_handler = logging.FileHandler('logfile.log')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)

    # add file handler to logger
    logger.addHandler(file_handler)
    main()
