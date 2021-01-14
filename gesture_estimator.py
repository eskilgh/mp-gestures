from cv2 import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import math
import numpy as np


# Dette burde endres, men er en lett måte å definere statiske gesturer
gestures = {
    "OPEN": [False] * 5,
    "CLOSED": [True] * 5,
    "PEACE": [True, False, False, True, True],
    "THUMBS_UP": [False, True, True, True, True],
    "POINTING": [True, False, True, True, True],
    "ROCK": [True, False, True, True, False],
}


# Kalkulering av hvilke ledd som hører til gitt finger
get_finger_joints = lambda finger: range(4 * finger + 1, 4 * (finger + 1))

# Magnituden til en vilkårlig dimensjonal vektor v
magnitude = lambda v: math.sqrt(sum(map(lambda X: math.pow(X, 2), v)))

# Inversen av dotpruktet
angle_between_vectors = lambda u, v: math.acos(
    np.divide(np.dot(u, v), magnitude(u) * magnitude(v))
)


class GestureEstimator:
    def __init__(self, landmarks: landmark_pb2.NormalizedLandmarkList):
        self.landmarks = landmarks.landmark
        self.set_finger_states()

    # Returnerer navnet til nåværende gestur
    # dersom en matchende gestur ikke blir funnet returnerer statusen til hver finger
    def get_gesture(self):

        # Prøver å finne en matchende gestur
        found = False
        for possible_gesture in gestures:
            if gestures[possible_gesture] == self.finger_states:
                gesture = possible_gesture
                found = True

        # Tilbakefall om match ikke blir funnet, lager en streng av finger_states
        fallback = lambda closed_fingers: "".join(
            [{False: "O", True: "C"}[f] for f in closed_fingers]
        )

        if found:
            return gesture
        else:
            return fallback(self.finger_states)

    # Tar inn indeksen til et ledd og returnerer krumning, et tall mellom 0 og 1/2 pi
    # Et ledd er et landemerke som ligger i mellom to andre landemerker
    # Altså en indeks mellom 4f + 1 til og med 4f + 3 for en finger f fra 0 til og med 4
    def joint_curvature(self, joint):
        # Henter koordinatene leddet og dens to nabonoder
        landmark_triplet = self.landmarks[joint - 1 : joint + 2].copy()

        # Dersom leddet er det første i en finger, settes første nabo til håndleddpunktet
        if (joint - 1) % 4 == 0:
            landmark_triplet[0] = self.landmarks[0]

        # Konverterer landemerkene til en liste av vektorer
        points = list(map(lambda p: np.array([p.x, p.y, p.z]), landmark_triplet))

        # Henter ut vektorene som peker fra leddet til begge nabopunktene
        u = points[0] - points[1]
        v = points[2] - points[1]

        # Finner vinkelen mellom vektorene, denne ligger mellom 0 og PI
        a = angle_between_vectors(u, v)

        return math.pi - a

    def finger_curvature(self, finger):
        return sum([self.joint_curvature(joint) for joint in get_finger_joints(finger)])

    def is_finger_closed(self, finger):
        return self.finger_curvature(finger) > math.pi / 2

    def set_finger_states(self):
        self.finger_states = [self.is_finger_closed(finger) for finger in range(5)]
