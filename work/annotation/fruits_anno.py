import json
import logging
import os
import os.path
from urllib.parse import urlparse
import numpy as np
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)

class GoogleLabels:

    def __init__(self):

        self.exclude_labels = {'Clemantine6.JPG': [[0.78, 0.02], [0.87, 0.03], [0.93, 0.12], [0.94, 0.20]],
                               'Fig2.JPG': [[0.94, 0.42], [0.95, 0.67]],
                               'Mandarin5.JPG': [[0.2, 0.04], [0.05, 0.06], [0.13, 0.05], [0.05, 0.35], [0.09, 0.39]]}




    def get_points(self, image, normalized_vertices, exclude_labels_normalized=None):
        height, width = image.shape[:2]

        points = np.array([[p['x'] if 'x' in p else 0, p['y'] if 'y' in p else 0] for p in normalized_vertices])

        points = np.round(points * [width, height]).astype(np.int32)

        # Checking if the label is excluded
        if exclude_labels_normalized is not None:

            exclude_labels = exclude_labels_normalized * np.array([width, height])

            if len(points) > 2:
                # Checking if exclude points are inside a polygon
                for exclude_point in exclude_labels:
                    # is_inside = cv2.pointPolygonTest(cnt, (50, 50), False)
                    if cv2.pointPolygonTest(points, tuple(exclude_point), False) >= 0:
                        return None  # The polygon should be excluded
            else:
                # Checking if exclude points are inside a rectangle
                for exclude_point in exclude_labels:
                    min_p = points.min(axis=0)
                    max_p = points.max(axis=0)
                    if min_p[0] <= exclude_point[0] <= max_p[0] and min_p[1] <= exclude_point[1] <= max_p[1]:
                        return None  # The rectangle should be excluded

        return points

    def draw_poly_bounding_rect(self, image, normalized_vertices, color, exclude_labels_normalized=None):

        points = self.get_points(image, normalized_vertices, exclude_labels_normalized)

        # mins = all_points.min(axis=0)
        # maxs = all_points.max(axis=0)

        if points is not None:
            cv2.rectangle(image, tuple(points.min(axis=0)), tuple(points.max(axis=0)), color, 2)

    def draw_poly(self, image, normalized_vertices, is_closed, color, exclude_labels_normalized=None):
        # height, width = image.shape[:2]
        #
        # # points = np.array([[(p['x'] if 'x' in p else 0) * width, (p['y'] if 'y' in p else 0) * height] for p in normalized_vertices])
        #
        # if height > width:
        #     # points = np.array([[(p['y'] if 'y' in p else 0) * width, (p['x'] if 'x' in p else 0) * height] for p in normalized_vertices])
        #     points = np.array([[p['y'] if 'y' in p else 0, p['x'] if 'x' in p else 0] for p in normalized_vertices])
        # else:
        #     # points = np.array([[(p['x'] if 'x' in p else 0) * width, (p['y'] if 'y' in p else 0) * height] for p in normalized_vertices])
        #     points = np.array([[p['x'] if 'x' in p else 0, p['y'] if 'y' in p else 0] for p in normalized_vertices])
        #
        # # points *= [width, height]
        #
        # # if height > width:
        # #     points = np.array([[(1 - (p['y'] if 'y' in p else 0)) * width, (p['x'] if 'x' in p else 0) * height] for p in
        # #                        normalized_vertices])
        # # else:
        # #     points = np.array([[(1 - (p['x'] if 'x' in p else 0)) * width, (1 - (p['y'] if 'y' in p else 0)) * height] for p in normalized_vertices])
        #
        # # points = np.array([[(p['x'] if 'x' in p else 0) * width, (p['y'] if 'y' in p else 0) * height] for p in normalized_vertices])
        # # points = np.array([[(p['x'] if 'x' in p else 0) * height, (p['y'] if 'y' in p else 0) * width] for p in normalized_vertices])
        # # points = np.array([[(1 - (p['y'] if 'y' in p else 0)) * width, (p['x'] if 'x' in p else 0) * height] for p in normalized_vertices])
        #
        # # points = np.array([[p['x'] * width, p['y'] * height] for p in normalized_vertices], dtype=np.int32)
        # # points = np.array([[p['x'] * height, p['y'] * width] for p in normalized_vertices], dtype=np.int32)
        #
        # points = np.round(points * [width, height]).astype(np.int32)

        points = self.get_points(image, normalized_vertices, exclude_labels_normalized)

        if points is not None:
            # cv2.polylines(image, [points], False, color, 2)  # not closed
            cv2.polylines(image, [points], is_closed, color, 2)
            # cv2.rectangle(image, tuple(points[0]), tuple(points[1]), color, 2)

    def draw_rectangle(self, image, normalized_vertices, color, exclude_labels_normalized=None):
        # height, width = image.shape[:2]
        #
        # # points = np.array([[(p['x'] if 'x' in p else 0) * width, (p['y'] if 'y' in p else 0) * height] for p in normalized_vertices])
        #
        # if height > width:
        #     points = np.array([[(p['y'] if 'y' in p else 0) * width, (p['x'] if 'x' in p else 0) * height] for p in
        #                        normalized_vertices])
        # else:
        #     points = np.array([[(p['x'] if 'x' in p else 0) * width, (p['y'] if 'y' in p else 0) * height] for p in normalized_vertices])
        # # if height > width:
        # #     points = np.array([[(1 - (p['y'] if 'y' in p else 0)) * width, (p['x'] if 'x' in p else 0) * height] for p in
        # #                        normalized_vertices])
        # # else:
        # #     points = np.array([[(1 - (p['x'] if 'x' in p else 0)) * width, (1 - (p['y'] if 'y' in p else 0)) * height] for p in normalized_vertices])
        #
        # # points = np.array([[(p['x'] if 'x' in p else 0) * width, (p['y'] if 'y' in p else 0) * height] for p in normalized_vertices])
        # # points = np.array([[(p['x'] if 'x' in p else 0) * height, (p['y'] if 'y' in p else 0) * width] for p in normalized_vertices])
        # # points = np.array([[(1 - (p['y'] if 'y' in p else 0)) * width, (p['x'] if 'x' in p else 0) * height] for p in normalized_vertices])
        #
        # # points = np.array([[p['x'] * width, p['y'] * height] for p in normalized_vertices], dtype=np.int32)
        # # points = np.array([[p['x'] * height, p['y'] * width] for p in normalized_vertices], dtype=np.int32)
        #
        # points = np.round(points).astype(np.int32)

        points = self.get_points(image, normalized_vertices, exclude_labels_normalized)

        # cv2.polylines(image, [points], False, color, 2)  # not closed

        # cv2.polylines(image, [points], True, color, 2)  # closed
        if points is not None and points.shape[0] == 2 and points.shape[1] == 2:
            cv2.rectangle(image, tuple(points[0]), tuple(points[1]), color, 2)

    def get_rectangle_sides_lengths(self, image, normalized_vertices):
        height, width = image.shape[:2]

        points = np.array([[(p['x'] if 'x' in p else 0) * width, (p['y'] if 'y' in p else 0) * height] for p in normalized_vertices])

        diff_square = (points - np.roll(points, -1, axis=0)) ** 2
        # lengths = np.sort(np.sqrt(np.sum(diff_square, axis=1)))

        lengths = np.sqrt(np.sum(diff_square, axis=1))
        return lengths

    def load_anno(self, anno_str=None):

        if anno_str == None:
            pass

            # with open('/Users/roman/work/ClariFruit/python/ML/ml_training/tomatoes_detection/Tomatoes_result1_partial.json', 'r') as anno_file:
            #     anno_str = anno_file.read()

        anno_dict = json.loads(anno_str)

        image_uri = anno_dict['image_payload']['image_uri']

        folder_name = 'D:/Clarifruit/cherry_stem/data/images_orig/'
        # file_name = urlparse(image_uri).path.replace('/MiscFruits/images/', folder_name)

        url_path = urlparse(image_uri).path
        file_name = os.path.basename(url_path)

        # if file_name != 'Mandarin5.JPG':
        #     return

        # local_image_path = os.path.join(folder_name, file_name)
        local_image_path = url_path.replace('/Cherry/images/', folder_name)
        # img = cv2.imread(os.path.join(folder_name, file_name))

        #img = Image(local_image_path)
        #img.prepare_for_detection()

        img = cv2.imread(local_image_path, cv2.IMREAD_UNCHANGED)

        image = img.copy()
        # image = img.original.copy()
        #image = img.resized.copy()



        all_fruits_dict = []
        num = 0

        for annotation in anno_dict['annotations']:

            ann_val = annotation['annotation_value']

            # Taking the first key name in the dictionary
            anno_type = next(iter(ann_val.keys()))

            normalized_vertices = ann_val[anno_type]['normalized_polyline']['normalized_vertices']
            # normalized_vertices = annotation['annotation_value']['image_polyline_annotation']['normalized_polyline']['normalized_vertices']

            label_class = ann_val[anno_type]['annotation_spec']['display_name']

            # if 'image_bounding_poly_annotation' in ann_val:
            #     normalized_vertices = annotation['annotation_value']['image_bounding_poly_annotation']['normalized_bounding_poly']['normalized_vertices']
            #     # normalized_vertices = annotation['annotation_value']['image_polyline_annotation']['normalized_polyline']['normalized_vertices']
            #
            #     label_class = annotation['annotation_value']['image_bounding_poly_annotation']['annotation_spec']['display_name']
            # elif 'image_polyline_annotation' in ann_val:
            #     normalized_vertices = annotation['annotation_value']['image_polyline_annotation']['normalized_polyline']['normalized_vertices']

            try:
                if anno_type == 'image_bounding_poly_annotation':
                    # draw_poly(image, normalized_vertices, True, (0, 255, 0))
                    self.draw_poly_bounding_rect(image, normalized_vertices, (0, 255, 0), self.exclude_labels.get(file_name))
                elif anno_type == 'image_polyline_annotation':
                    self.draw_poly(image, normalized_vertices, False, (0, 255, 0), self.exclude_labels.get(file_name))

            except Exception as e:
                logger.error('Cannot draw annotation!', exc_info=1)
                # logger.exception()

            # try:
            #     if label_class == 'fruit':
            #         draw_poly(image, normalized_vertices, (0, 255, 0))
            #     if label_class == 'white ball':
            #         draw_rectangle(image, normalized_vertices, (0, 255, 0))
            #
            # except Exception as e:
            #     logger.error('Cannot draw polygon!')
            #     # raise


        save_folder = 'D:/Clarifruit/cherry_stem/data/anno_images'
        # save_results_folder = '/Volumes/Samsung_T5/data/test_ref_images/labels_by_google1_results/'

        # cv2.imshow(file_name, image)
        cv2.imwrite(os.path.join(save_folder, file_name), image)

    def save_all_anno_images(self):

        with open('D:/Clarifruit/cherry_stem/data/Cherry_cherry_stem_result.json', 'r') as anno_file:
            anno_str_list = anno_file.readlines()

        # with open('/Volumes/Samsung_T5/data/misc_fruits/MiscFruits_misc_fruits_ball.json', 'r') as anno_file:
        #     anno_str_list = anno_file.readlines()

        for anno_str in tqdm(anno_str_list):
            self.load_anno(anno_str)
            # break


    # def get_images_without_anno():
    #
    #     from glob import glob
    #
    #     all_files = glob('/Volumes/Samsung_T5/data/tomatoes_detection/images/*.jpg')
    #     anno_files = glob('/Volumes/Samsung_T5/data/tomatoes_detection/images_anno/*.jpg')
    #
    #     all_file_names = set([os.path.basename(file_path) for file_path in all_files])
    #     anno_file_names = set([os.path.basename(file_path) for file_path in anno_files])
    #
    #     diff = all_file_names - anno_file_names


