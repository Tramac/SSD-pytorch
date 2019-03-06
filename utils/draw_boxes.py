"""Draw predicted or ground truth boxes on input image."""
import colorsys
import random
import cv2
import numpy as np


def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)
    random.shuffle(colors)
    random.seed(None)

    return colors


colors = get_colors_for_classes(21)


def draw_boxes(image, box, box_txt, color_index):
    """Draw bounding boxes on image.
    image: np.array
    box: [left, top, right, bottom]
    box_txt: (class: score)
    """
    h, w = image.shape[:2]
    thickness = (h + w) // 300
    left, top, right, bottom = box
    top = max(0, np.round(top).astype(np.int32))
    left = max(0, np.around(left).astype(np.int32))
    right = min(w, np.round(right).astype(np.int32))
    bottom = min(h, np.round(bottom).astype(np.int32))
    cv2.rectangle(image, (left, top), (right, bottom), colors[color_index], thickness)
    cv2.putText(image, box_txt, (left, top - 5), 0, 0.8, colors[color_index], 2)
