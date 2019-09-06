import json
import logging
import os
import os.path
from urllib.parse import urlparse
import numpy as np
import cv2
from tqdm import tqdm
import csv
import shutil

logger = logging.getLogger(__name__)

class GoogleLabels:

    def __init__(self, anno_path,csv_path, src_images_path, dest_path, is_mask=False):
        self.anno_path = anno_path
        self.csv_path = csv_path
        self.src_image_path = src_images_path
        self.dest_path = dest_path
        self.is_mask = is_mask


        # exclude annotations for specific images
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


        if points is not None:
            cv2.rectangle(image, tuple(points.min(axis=0)), tuple(points.max(axis=0)), color, 2)

    def draw_poly(self, image, normalized_vertices, is_closed, color, exclude_labels_normalized=None):


        points = self.get_points(image, normalized_vertices, exclude_labels_normalized)

        if points is not None:

            cv2.polylines(image, [points], is_closed, color, 2)


    def draw_rectangle(self, image, normalized_vertices, color, exclude_labels_normalized=None):


        points = self.get_points(image, normalized_vertices, exclude_labels_normalized)


        if points is not None and points.shape[0] == 2 and points.shape[1] == 2:
            cv2.rectangle(image, tuple(points[0]), tuple(points[1]), color, 2)

    def get_rectangle_sides_lengths(self, image, normalized_vertices):
        height, width = image.shape[:2]

        points = np.array([[(p['x'] if 'x' in p else 0) * width, (p['y'] if 'y' in p else 0) * height] for p in normalized_vertices])

        diff_square = (points - np.roll(points, -1, axis=0)) ** 2

        lengths = np.sqrt(np.sum(diff_square, axis=1))
        return lengths

    def load_anno(self, anno_str=None):

        if anno_str == None:
            pass

        anno_dict = json.loads(anno_str)

        image_uri = anno_dict['image_payload']['image_uri']

        url_path = urlparse(image_uri).path
        file_name = os.path.basename(url_path)

        local_image_path = url_path.replace('/Cherry/images/', self.src_image_path)

        img = cv2.imread(local_image_path, cv2.IMREAD_UNCHANGED)

        if self.is_mask:
            height, width = img.shape[:2]
            image = np.zeros((height, width, 1), np.uint8)
            mask_color = 255
        else:
            image = img.copy()
            mask_color = (0,255,0)

        for annotation in anno_dict['annotations']:

            ann_val = annotation['annotation_value']

            # Taking the first key name in the dictionary
            anno_type = next(iter(ann_val.keys()))

            normalized_vertices = ann_val[anno_type]['normalized_polyline']['normalized_vertices']

            try:
                if anno_type == 'image_bounding_poly_annotation':
                    # draw_poly(image, normalized_vertices, True, (0, 255, 0))
                    self.draw_poly_bounding_rect(image, normalized_vertices, mask_color, self.exclude_labels.get(file_name))
                elif anno_type == 'image_polyline_annotation':
                    self.draw_poly(image, normalized_vertices, False, mask_color, self.exclude_labels.get(file_name))

            except Exception:
                logger.error('Cannot draw annotation!', exc_info=1)

        cv2.imwrite(os.path.join(self.dest_path, file_name), image)
        return local_image_path

    def get_from_csv(self):
        res = []
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                url_path = urlparse(row[0]).path

                local_image_path = url_path.replace('/Cherry/images/', self.src_image_path)
                res.append(local_image_path)
        return res

    def make_blank_masks(self, imgs_list):
        for file in imgs_list:
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            height, width = img.shape[:2]
            image = np.zeros((height, width, 1), np.uint8)
            file_name = os.path.basename(file)
            cv2.imwrite(os.path.join(self.dest_path, file_name), image)

    def move_anno_images(self,img_list):
        train_path = os.path


    def save_all_anno_images(self):
        with open(self.anno_path, 'r') as ano_file:
            anno_str_list = ano_file.readlines()


        anno_paths = set()
        for anno_str in tqdm(anno_str_list):
            anno_paths.add(self.load_anno(anno_str))

        images_list = set(self.get_from_csv())
        train_path = r'D:\Clarifruit\cherry_stem\data\train'
        for img in images_list:
            _ = shutil.copy(img, train_path)


        no_anno = images_list.difference(anno_paths)
        self.make_blank_masks(no_anno)






