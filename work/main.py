from work.annotation import fruits_anno
import csv
import os
from urllib.parse import urlparse

def annotate():
    ano_path = r'D:\Clarifruit\cherry_stem\data\Cherry_cherry_stem_result.json'
    src_images_path = r'D:/Clarifruit/cherry_stem/data/images_orig/'
    dest_path = r'D:/Clarifruit/cherry_stem/data/masks/'
    csv_path = r'D:\Clarifruit\cherry_stem\data\cherry_stem.csv'


    gl = fruits_anno.GoogleLabels(anno_path=ano_path,
                                  src_images_path=src_images_path,
                                  csv_path=csv_path,
                                  dest_path=dest_path,
                                  is_mask=True)

    gl.save_all_anno_images()


def main():
    pass


if __name__ == "__main__":
    main()
