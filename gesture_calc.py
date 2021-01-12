import numpy as np
import mediapipe as mp
from cv2 import cv2

FINGER_INDICES = {
    # "THUMB": [1, 2, 3, 4],
    "INDEX": [5, 6, 7, 8],
    "MIDDLE": [9, 10, 11, 12],
    "RING": [13, 14, 15, 16],
    "PINKY": [17, 18, 19, 20],
}


def sq_distance(a, b):
    return (b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2


class GestureCalculator:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.set_finger_states()

    def process(self):
        if self.hand_is_closed():
            return "FIST"
        elif self.is_peace_sign():
            return "PEACE"
        elif self.is_pointing():
            return "POINT"
        elif self.is_rocking():
            return "ROCK!"
        return None

    def set_finger_states(self):
        self.finger_states = {"THUMB": "OPEN" if self.thumb_is_open() else "CLOSED"}
        print("thumb is: ", self.finger_states["THUMB"])
        for finger in FINGER_INDICES:
            self.finger_states[finger] = (
                "OPEN" if self.finger_is_open(finger) else "CLOSED"
            )

    def hand_is_closed(self):
        return (
            # self.finger_states["THUMB"] is "CLOSED"
            self.finger_states["INDEX"] is "CLOSED"
            and self.finger_states["MIDDLE"] is "CLOSED"
            and self.finger_states["RING"] is "CLOSED"
            and self.finger_states["PINKY"] is "CLOSED"
        )

    def is_peace_sign(self):
        return (
            # self.finger_states["THUMB"] is "CLOSED"
            self.finger_states["INDEX"] is "OPEN"
            and self.finger_states["MIDDLE"] is "OPEN"
            and self.finger_states["RING"] is "CLOSED"
            and self.finger_states["PINKY"] is "CLOSED"
        )

    def is_pointing(self):
        return (
            self.finger_states["INDEX"] is "OPEN"
            and self.finger_states["MIDDLE"] is "CLOSED"
            and self.finger_states["RING"] is "CLOSED"
            and self.finger_states["PINKY"] is "CLOSED"
        )

    def is_rocking(self):
        return (
            self.finger_states["INDEX"] is "OPEN"
            and self.finger_states["MIDDLE"] is "CLOSED"
            and self.finger_states["RING"] is "CLOSED"
            and self.finger_states["PINKY"] is "OPEN"
        )

    def finger_is_open(self, finger):
        """Returns:
            True if distance from TIP to WRIST is greater than distance from IP to WRIST
        """
        WRIST = self.landmarks[0]
        DIP = self.landmarks[FINGER_INDICES[finger][2]]
        TIP = self.landmarks[FINGER_INDICES[finger][3]]
        return sq_distance(WRIST, TIP) > sq_distance(WRIST, DIP)

    def thumb_is_open(self):
        WRIST = self.landmarks[0]
        THUMB_CMC = self.landmarks[1]
        THUMB_MCP = self.landmarks[2]
        THUMB_IP = self.landmarks[3]
        THUMB_TIP = self.landmarks[4]
        INDEX_MCP = self.landmarks[5]
        return sq_distance(THUMB_IP, INDEX_MCP) > sq_distance(THUMB_MCP, THUMB_IP)

