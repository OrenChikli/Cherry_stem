import logging
import base64
import cv2
import numpy as np
from itertools import compress
from work.segmentation.common import Common

logger = logging.getLogger(__name__)


class Utils:

    @staticmethod
    def imshow(winname, image, img_log_levels_list):

        if Common.imgLogLevel in img_log_levels_list:
            cv2.imshow(winname, image)

    @staticmethod
    def count_pixels_in_circle_and_mask(circle, mask):
        return np.count_nonzero(Utils.get_circle_and_mask_intersection(circle, mask))

    @staticmethod
    def get_circle_color_with_mask(image, circle, channel, mask=None):
        values = Utils.get_circle_values_with_mask(image, circle, mask)

        if channel is None:
            mean_color = values.mean(axis=0)
            var_color = values.var(axis=0)
        else:
            channel_values = values[:, channel]
            mean_color = channel_values.mean()
            var_color = channel_values.var()

        return values.shape[0], mean_color, var_color

    @staticmethod
    def get_circle_angular_color_with_mask(image, circle, channel, mask=None):
        values = Utils.get_circle_values_with_mask(image, circle, mask)

        if len(values) == 0:
            return 0, np.nan, np.nan

        if channel is None:
            mean_color = values.mean(axis=0)
            var_color = values.var(axis=0)
        else:
            channel_values = values[:, channel]
            mean_color, var_color = Utils.calc_hue_mean_and_var(channel_values)

        return values.shape[0], mean_color, var_color

    @staticmethod
    def create_circle_mask(shape, circle):
        # The mask is an image containing the values 0 and 255
        circle_mask = np.zeros(shape, dtype=np.uint8)

        # drawing a circle.
        circle = np.rint(circle).astype(np.int)
        cv2.circle(circle_mask, (circle[0], circle[1]), circle[2], 255, -1)

        return circle_mask

    @staticmethod
    def get_circle_and_mask_intersection(circle, mask):
        # The mask is an image containing the values 0 and 255
        return (Utils.create_circle_mask(mask.shape, circle) > 0) * (mask > 0)

    @staticmethod
    def get_circle_values_with_mask(image, circle, mask):
        # The mask is an image containing the values 0 and 255

        # Ignoring values outside of the circle and not in the mask if mask is provided.
        if mask is None:
            values = image[(Utils.create_circle_mask(image.shape[0:2], circle) > 0)]
        else:
            values = image[Utils.get_circle_and_mask_intersection(circle, mask)]
        return values

    @staticmethod
    def calc_hue_mean_and_var(vector, weights=None):
        # vector range is (0 - 180) (hue)
        mean_color, var = Utils.calc_angular_mean_and_var(vector.astype(np.uint16) * 2, weights)
        return mean_color / 2., var

    @staticmethod
    def calc_angular_mean_and_var(vector, weights=None):

        if len(vector) == 0:
            return np.nan, np.nan

        # vector range is (0 - 360)
        # angles1 = vector.astype(np.float) * (np.pi / 180.)
        angles = np.deg2rad(vector)

        if weights is None or len(weights) != len(vector):
            x = np.cos(angles).sum()
            y = np.sin(angles).sum()
            n = vector.size
        else:
            x = (np.cos(angles) * weights).sum()
            y = (np.sin(angles) * weights).sum()
            n = weights.sum()

        a = np.arctan2(y, x)
        # the return value of arctan2 is in the range [-pi, pi].
        if a < 0:
            a += 2. * np.pi
        # mean1 = a * (180. / np.pi)
        mean = np.rad2deg(a)

        r = np.sqrt(x * x + y * y) / n
        var = 1 - r

        return mean, var

    # Resize the image so the longer side will be 768 pixels.
    @staticmethod
    def resize_image(image, resized_image_long_side):
        height, width = image.shape[:2]

        if height > width:
            f = float(resized_image_long_side) / height
        else:
            f = float(resized_image_long_side) / width

        if f == 1.0:
            # No scaling required
            return image, 1.0

        # smallImage = cv2.resize(image, (0,0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
        resized_image = cv2.resize(image, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_AREA)
        # smallImage = cv2.resize(image, (640, 480), interpolation = cv2.INTER_AREA)
        return resized_image, f

    @staticmethod
    def compute_edges(image):
        img_blurred = cv2.GaussianBlur(image, (5, 5), 2, 2)
        high_thresh, thresh_im = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_thresh /= 2.
        low_thresh = 0.5 * high_thresh

        # sigma = 0.33
        # v = np.median(img_blurred)
        # lower = int(max(0, (1.0 - sigma) * v))
        # upper = int(min(255, (1.0 + sigma) * v))

        img_canny = cv2.Canny(img_blurred, low_thresh, high_thresh)

        # count = np.count_nonzero(img_canny)
        # size = image.shape[0] * image.shape[1]

        # c = count / size
        # logger.debug('edges ratio: %f', c)

        return img_canny
        # return Utils.auto_canny(img_blurred)

    # @staticmethod
    # def auto_canny(image, sigma=0.33):
    #     # compute the median of the single channel pixel intensities
    #     v = np.median(image)
    #
    #     # apply automatic Canny edge detection using the computed median
    #     lower = int(max(0, (1.0 - sigma) * v))
    #     upper = int(min(255, (1.0 + sigma) * v))
    #     edged = cv2.Canny(image, lower, upper)
    #
    #     # return the edged image
    #     return edged

    @staticmethod
    def check_circles_touch(c1, c2):
        return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 <= (c1[2] + c2[2]) ** 2

    @staticmethod
    def check_point_inside_circle(circle, x, y):
        return (circle[0] - x) ** 2 + (circle[1] - y) ** 2 <= circle[2] ** 2

    @staticmethod
    def round_to_int(value):
        return int(round(value))

    @staticmethod
    def same_side_of_line(line_point1, line_point2, p1, p2):
        l1x = line_point1[1]
        l1y = line_point1[0]

        l2x = line_point2[1]
        l2y = line_point2[0]

        p1x = p1[1]
        p1y = p1[0]

        p2x = p2[1]
        p2y = p2[0]

        return ((l1y-l2y) * (p1x-l1x) + (l2x-l1x) * (p1y-l1y)) * \
               ((l1y-l2y) * (p2x-l1x) + (l2x-l1x) * (p2y-l1y)) > 0

    @staticmethod
    def square_distance(p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    @staticmethod
    def crop_image_by_rect(image, rect, expand_ratio):

        # if expand_ratio is 1, the size will not change

        top, left, bottom, right = rect
        height = (bottom - top) * expand_ratio
        width = (right - left) * expand_ratio

        # center = [(bottom + top) / 2., (right + left) / 2.]

        img_height, img_width = image.shape[:2]
        top = max(0, int(round(top - (expand_ratio - 1) * height / 2.)))
        left = max(0, int(round(left - (expand_ratio - 1) * width / 2.)))
        bottom = min(img_height - 1, int(round(bottom + (expand_ratio - 1) * height / 2.)))
        right = min(img_width - 1, int(round(right + (expand_ratio - 1) * width / 2.)))

        crop_image = image[top:bottom, left:right]

        return crop_image, (left, top)

    @staticmethod
    def crop_image(circle, image):
        height, width = image.shape[:2]

        if width == 0 or height == 0:
            return None, None

        crop_radius = circle[2]
        crop_left = Utils.round_to_int(circle[0] - crop_radius)
        if crop_left < 0:
            crop_left = 0
        crop_top = Utils.round_to_int(circle[1] - crop_radius)
        if crop_top < 0:
            crop_top = 0
        crop_right = Utils.round_to_int(circle[0] + crop_radius)
        if crop_right >= width:
            crop_right = width - 1
        crop_bottom = Utils.round_to_int(circle[1] + crop_radius)
        if crop_bottom >= height:
            crop_bottom = height - 1
        crop_img = image[crop_top:crop_bottom, crop_left:crop_right]

        return crop_img, (crop_left, crop_top)

    @staticmethod
    def remove_outliers_with_data(circles, value):

        if circles is None:
            return None

        # No outliers for less then 2 items
        if 0 <= len(circles) <= 2:
            return circles

        radii = [circle[2] for (circle, _) in circles]
        filt = abs(radii - np.mean(radii)) <= value * np.std(radii)
        return list(compress(circles, filt))

    @staticmethod
    def remove_outliers(circles, value):

        if circles is None:
            return None

        # No outliers for less then 2 items
        if 0 <= len(circles) <= 2:
            return circles

        radii = circles[:, 2]
        return circles[abs(radii - np.mean(radii)) <= value * np.std(radii)]

    @staticmethod
    def remove_outliers_in_vector_filter(vector, low_value, high_value):

        if vector is None:
            return None

        # No outliers for less then 2 items
        if 0 <= len(vector) <= 2:
            return vector

        std = np.std(vector)

        # np.sort(vector[np.median(vector) - vector <= 1 * np.std(vector) * vector - np.median(vector) <= 2 * np.std(vector)])

        # return vector[abs((np.median(vector) + (high_value - low_value) / 2. * std) - vector) <= (low_value + high_value) / 2. * std]
        # return vector[(np.median(vector) - vector <= low_value * np.std(vector)) * (vector - np.median(vector) <= high_value * np.std(vector))]
        return abs((np.median(vector) + (high_value - low_value) / 2. * std) - vector) <= (low_value + high_value) / 2. * std

    @staticmethod
    def mark_angular_outliers(vector, value=1):
        # vector must have values [0 - 360]
        # Returns bool array where True - inlier, False - outlier
        vector_mean, vector_var = Utils.calc_angular_mean_and_var(vector)
        var_for_outliers = pow(1 - vector_var, value)

        # calculating the maximal angle from mean for the sample not to be outlier
        angle_for_outliers = 2. * np.rad2deg(np.arccos(var_for_outliers))

        # diff_vector = 180 - abs(abs(vector_mean - vector) - 180)  # calculates the distance angle between vector and vector_mean
        # result = diff_vector <= angle_for_outliers
        # optimized for less calculations
        return 180 - angle_for_outliers <= abs(abs(vector_mean - vector) - 180)
        # return np.array([Utils.calc_angular_mean_and_var(np.array([vector_mean, value], np.float))[1] <= 1 - var_for_outliers for value in vector], np.bool)

    @staticmethod
    def remove_angular_outliers(vector, value=1):
        return vector[Utils.mark_angular_outliers(vector, value)]

    @staticmethod
    def remove_hue_outliers(hue_values, value=1):
        return hue_values[Utils.mark_angular_outliers(hue_values.astype(np.uint16) * 2, value)]

    @staticmethod
    def are_values_close(v1, v2, ratio):
        max_side = max(v1, v2)
        min_side = min(v1, v2)

        rect_ratio = min_side / max_side
        return rect_ratio >= ratio

    @staticmethod
    def are_circles_similar(circle1, circle2):

        # Checking that the radii of both circles are 90% close and
        # that the distance between the centers is less then 10% percent of the smallest radius.
        return Utils.are_values_close(circle1[2], circle2[2], 0.9) and \
               Utils.square_distance(circle1[:2], circle2[:2]) <= (min(circle1[2], circle2[2]) * 0.1) ** 2

    @staticmethod
    def is_similar_circle_in_list(circle, circles_list):

        if circle is None or circles_list is None:
            return False

        for c in circles_list:
            if Utils.are_circles_similar(circle, c):
                return True

        return False

    @staticmethod
    def circle_to_json(circle):
        return {'x': circle[0], 'y': circle[1], 'radius': circle[2]}

    @staticmethod
    def array_from_iterable(iterable, key):
        arr = np.array([item[key] for item in iterable if key in item])
        return arr[arr != np.array([None])]
        # return np.array([item[key] for item in iterable])

    @staticmethod
    def array_from_list_of_dict_by_key(list_of_dict, key):
        return np.array([dictionary[key] for dictionary in list_of_dict if key in dictionary])

    @staticmethod
    def base64_encode(image, ext='.jpg'):
        retval, buffer = cv2.imencode(ext, image)
        if retval:
            image_str = base64.b64encode(buffer)
            return image_str
        else:
            return None

    @staticmethod
    def base64_decode(image_str):
        decoded_buffer = base64.b64decode(image_str)
        image_np = np.frombuffer(decoded_buffer, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_np, flags=cv2.IMREAD_UNCHANGED)
        return decoded_image

    @staticmethod
    def pad_image_for_rotation(image):

        h, w = image.shape[:2]
        # size = int(round(np.sqrt(h ** 2 + w ** 2) / 2))

        size = int(round(np.sqrt(h ** 2 + w ** 2)))

        # new_image = np.zeros([size, size], dtype=np.uint8)

        nh = int(round((size - h) / 2.))
        nw = int(round((size - w) / 2.))
        nw1 = h + 2 * nh - w - nw  # To make sure we get a square image

        new_image = cv2.copyMakeBorder(image, nh, nh, nw, nw1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_image, np.array([nh, nw])

    @staticmethod
    def rotate_image(image, angle):

        (h, w) = image.shape[:2]
        # (cX, cY) = (w // 2, h // 2)
        (cX, cY) = (w / 2., h / 2.)
        matrix = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        return cv2.warpAffine(image, matrix, (h, w))

    @staticmethod
    def rotate_coords(points, center, angle):

        # pr0 = rotate_coords0(point, center, angle)

        # cy, cx = center  # flip x and y
        # We rotate by -angle instead of angle because the coordinates are flipped x <-> y !!
        matrix = cv2.getRotationMatrix2D(tuple(center), -angle, 1.0)
        pr = cv2.transform(points, matrix)
        return pr

    @staticmethod
    def get_circles_inside_rotated_box(rotated_box):

        side_lengths = Utils.get_rotated_boxes_sides_lengths(rotated_box)
        radius = np.min(side_lengths) / 2.

        # Creating an affine transformation from the box to a rectangle that lies on the origin and the axes, same proportions
        tr_side = side_lengths[0] / side_lengths[1]
        M = cv2.getAffineTransform(np.float32([[tr_side, 0], [0, 0], [0, 1]]), rotated_box[:3].astype(np.float32))

        tr_radius = min(tr_side, 1) / 2.

        center1 = np.float32([tr_radius, tr_radius])
        if tr_side > 1:
            center2 = np.float32([tr_side - tr_radius, tr_radius])
        else:
            center2 = np.float32([tr_radius, 1 - tr_radius])

        centers = np.fliplr(cv2.transform(np.array([np.stack([center1, center2])]), M)[0])

        circles = np.append(centers, np.array([[radius], [radius]]), axis=1)

        return circles

    @staticmethod
    def get_rotated_boxes_sides_lengths(boxes):
        points = boxes
        diff_square = (points - np.roll(points, -1, axis=-2)) ** 2
        # lengths = np.sort(np.sqrt(np.sum(diff_square, axis=1)))

        lengths = np.sqrt(np.sum(diff_square, axis=-1))
        return lengths

    @staticmethod
    def draw_rotated_boxes(image, rotated_boxes, color):
        points = np.round(rotated_boxes).astype(np.int32)
        # points_xy = points[..., ::-1].copy()
        points_xy = np.ascontiguousarray(points[..., ::-1])
        cv2.polylines(image, points_xy, True, color, 2)

    @staticmethod
    def draw_text(img, text, origin):
        thickness = 2
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1.
        textsize, baseline = cv2.getTextSize(text, fontface, fontscale, thickness)
        baseline += thickness
        text_origin = np.array((origin[0], origin[1] - textsize[1]))
        cv2.rectangle(img, tuple((text_origin + (0, baseline)).astype(int)),
                      tuple((text_origin + (textsize[0], -textsize[1])).astype(int)), (128, 128, 128), -1)
        cv2.putText(img, text, tuple((text_origin + (0, baseline / 2)).astype(int)), fontface, fontscale,
                    (0, 0, 0), thickness, 8)
        return img
