import math
import numpy as np
from cv2 import cv2
from typing import List, Tuple, Union
from mediapipe.framework.formats import landmark_pb2

RED_COLOR = (0, 0, 255)

stroke = []


def draw_handmarks_label(
    img: np.ndarray,
    text: str,
    hand_landmarks: landmark_pb2.NormalizedLandmarkList,
    margin=20,
):
    img_rows, img_cols, _ = img.shape
    points = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
    x_min, y_min, _, _ = get_edges_in_pixels(points, img_cols, img_rows)
    pos = (x_min, (y_min - margin))
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2

    cv2.putText(img, text, pos, font_face, scale, RED_COLOR, 1, cv2.LINE_AA)


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def get_edges_in_pixels(points: List[Tuple[int, int]], img_width: int, img_height: int):
    x_min = x_max = points[0][0]
    y_min = y_max = points[0][1]
    for x, y in points:
        x_min = x if x < x_min else x_min
        x_max = x if x > x_max else x_max
        y_min = y if y < y_min else y_min
        y_max = y if y > y_max else y_max
    return normalized_to_pixel_coordinates(
        x_min, y_min, img_width, img_height
    ) + normalized_to_pixel_coordinates(x_max, y_max, img_width, img_height)


def draw_landmark_bbox(
    img: np.ndarray, hand_landmarks: landmark_pb2.NormalizedLandmarkList
):
    img_rows, img_cols, _ = img.shape
    points = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
    x_min, y_min, x_max, y_max = get_edges_in_pixels(points, img_cols, img_rows)
    cv2.rectangle(img, (x_min, y_max), (x_max, y_min), RED_COLOR)

# Viser malekosten og legger til i tegning dersom paint er sann 
def show_brush(img: np.ndarray, fingertip: np.ndarray, paint: bool):
    x, y = calculate_stroke(img, fingertip)
    cv2.ellipse(img, (x, y), (20, 20), 0, 0, 360, 222, 10)

    if paint:
        stroke.append((x, y))


# Tegner linjer mellom alle punkter lagret i tegning
def show_paint(img: np.ndarray):
    for i in range(len(stroke) - 1):
        cv2.line(img, stroke[i], stroke[i + 1], 222, 5)

# Syntes normalized_fingertip tydeligvis var noe lang å skrive ut
# Lagde denne helt på automatikk
def calculate_stroke(img: np.ndarray, fingertip: np.ndarray):
    img_height, img_width, _ = img.shape

    normalized_fingertip = normalized_to_pixel_coordinates(
        fingertip[0], fingertip[1], img_width, img_height
    )

    return (normalized_fingertip[0], normalized_fingertip[1])
