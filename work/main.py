from work.annotation import fruits_anno
import csv
import os
from urllib.parse import urlparse





def main():
    ano_path = 'D:/Clarifruit/cherry_stem/data/Cherry_cherry_stem_result.json'
    src_images_path = 'D:/Clarifruit/cherry_stem/data/images_orig/'
    dest_path = 'D:/Clarifruit/cherry_stem/data/masks/'
    csv_path = 'D:\Clarifruit\cherry_stem\data\cherry_stem.csv'


    gl = fruits_anno.GoogleLabels(anno_path=ano_path,
                                  src_images_path=src_images_path,
                                  csv_path=csv_path,
                                  dest_path=dest_path,
                                  is_mask=True)

    gl.save_all_anno_images()

if __name__ == "__main__":
    main()
