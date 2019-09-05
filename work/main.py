from .annotation import fruits_anno
def main():

    gl = fruits_anno.GoogleLabels()

    # get_images_without_anno()

    # load_anno()
    gl.save_all_anno_images()

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()