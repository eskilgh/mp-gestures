import mediapipe as mp
import numpy as np
import math
from cv2 import cv2
from typing import List, Tuple, Union
from mediapipe.framework.formats import landmark_pb2

RED_COLOR = (0, 0, 255)


def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 4
    color = (0, 0, 0)
    # thickness = cv2.LINE_AA
    margin = 2

    # txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    # end_x = pos[0] + txt_size[0][0] + margin
    # end_y = pos[1] - txt_size[0][1] - margin

    # cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, RED_COLOR, 1, cv2.LINE_AA)


def draw_handmarks_label(img, text, hand_landmarks, margin=20):
    img_rows, img_cols, _ = img.shape
    points = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
    x_min, y_min, _, _ = _get_edges_in_pixels(points, img_cols, img_rows)
    pos = x_min, y_min - margin
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2

    # txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    # end_x = pos[0] + txt_size[0][0] + margin
    # end_y = pos[1] - txt_size[0][1] - margin

    # cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, RED_COLOR, 1, cv2.LINE_AA)


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def _get_edges_in_pixels(points, img_width, img_height):
    x_min = x_max = points[0][0]
    y_min = y_max = points[0][1]
    for x, y in points:
        x_min = x if x < x_min else x_min
        x_max = x if x > x_max else x_max
        y_min = y if y < y_min else y_min
        y_max = y if y > y_max else y_max
    return _normalized_to_pixel_coordinates(
        x_min, y_min, img_width, img_height
    ) + _normalized_to_pixel_coordinates(x_max, y_max, img_width, img_height)


def draw_landmark_bbox(
    img: np.ndarray, hand_landmarks: landmark_pb2.NormalizedLandmarkList
):
    img_rows, img_cols, _ = img.shape
    points = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
    x_min, y_min, x_max, y_max = _get_edges_in_pixels(points, img_cols, img_rows)
    cv2.rectangle(img, (x_min, y_max), (x_max, y_min), RED_COLOR)


class GestureClassifier:
    def __init__(self, multi_hand_landmarks):
        self.multi_hand_landmarks = multi_hand_landmarks

    def process(self):
        for landmarks in self.multi_hand_landmarks:
            if self.hand_is_closed(landmarks.landmark):
                return 1
            elif self.is_peace_sign(landmarks.landmark):
                return 2
        return 0

    def thumb_is_open(self, hand_landmarks):
        """Returns:
            True if distance from THUMB_TIP to WRIST is greater than distance from THUMB_IP to WRIST
        """
        WRIST = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
        THUMB_IP = np.array([hand_landmarks[3].x, hand_landmarks[3].y])
        THUMB_TIP = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
        return np.linalg.norm(THUMB_TIP - WRIST) > np.linalg.norm(THUMB_IP - WRIST)

    def finger_is_open(self, finger_landmarks):
        MCP = finger_landmarks[0]
        PIP = finger_landmarks[1]
        DIP = finger_landmarks[2]
        TIP = finger_landmarks[3]
        return TIP.y < PIP.y

    def split_finger_landmarks(self, hand_landmarks):
        return [
            hand_landmarks[i : i + 4] for i in list(range(len(hand_landmarks)))[5::4]
        ]

    def hand_is_closed(self, hand_landmarks):
        for finger_landmarks in self.split_finger_landmarks(hand_landmarks):
            if self.finger_is_open(finger_landmarks):
                return False
        return not self.thumb_is_open(hand_landmarks)

    def is_peace_sign(self, hand_landmarks):
        finger_landmarks = self.split_finger_landmarks(hand_landmarks)
        return (
            # not self.thumb_is_open(hand_landmarks)
            self.finger_is_open(finger_landmarks[0])
            and self.finger_is_open(finger_landmarks[1])
            and not self.finger_is_open(finger_landmarks[2])
            and not self.finger_is_open(finger_landmarks[3])
        )

