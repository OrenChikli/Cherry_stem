from work.annotation import fruits_anno
import glob

def main():
    ano_path = 'D:/Clarifruit/cherry_stem/data/Cherry_cherry_stem_result.json'
    src_images_path = 'D:/Clarifruit/cherry_stem/data/images_orig/'
    dest_path = 'D:/Clarifruit/cherry_stem/data/masks/'



    gl = fruits_anno.GoogleLabels(ano_path, src_images_path, dest_path, True)

    gl.save_all_anno_images()

if __name__ == "__main__":
    main()
