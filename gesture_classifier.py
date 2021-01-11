import mediapipe as mp
import numpy as np
from cv2 import cv2


def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


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

