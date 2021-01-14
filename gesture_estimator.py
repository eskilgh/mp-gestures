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
getFingerJoints = lambda finger: range(4 * finger + 1, 4 * (finger + 1))

# Magnituden til en vilkårlig dimensjonal vektor v
magnitude = lambda v: math.sqrt(sum(map(lambda X: math.pow(X, 2), v)))

# Inversen av dotpruktet
angleBetweenVectors = lambda u, v: math.acos(
    np.divide(np.dot(u, v), magnitude(u) * magnitude(v))
)


class GestureEstimator:
    def __init__(self, landmarks: landmark_pb2.NormalizedLandmarkList):
        self.landmarks = landmarks.landmark
        self.set_finger_states()

    # Returnerer navnet til nåværende gestur
    # dersom en matchende gestur ikke blir funnet returnerer statusen til hver finger
    def getGesture(self):

        # Prøver å finne en matchende gestur
        found = False
        for possibleGesture in gestures:
            if gestures[possibleGesture] == self.finger_states:
                gesture = possibleGesture
                found = True

        # Tilbakefall om match ikke blir funnet, lager en streng av finger_states
        fallback = lambda closedFingers: "".join(
            [{False: "O", True: "C"}[f] for f in closedFingers]
        )

        if found:
            return gesture
        else:
            return fallback(self.finger_states)

    # Tar inn indeksen til et ledd og returnerer krumning, et tall mellom 0 og 1/2 pi
    # Et ledd er et landemerke som ligger i mellom to andre landemerker
    # Altså en indeks mellom 4f + 1 til og med 4f + 3 for en finger f fra 0 til og med 4
    def jointCurvature(self, joint):
        # Henter koordinatene leddet og dens to nabonoder
        landmarkTriplet = self.landmarks[joint - 1 : joint + 2].copy()

        # Dersom leddet er det første i en finger, settes første nabo til håndleddpunktet
        if (joint - 1) % 4 == 0:
            landmarkTriplet[0] = self.landmarks[0]

        # Konverterer landemerkene til en liste av vektorer
        points = list(map(lambda p: np.array([p.x, p.y, p.z]), landmarkTriplet))

        # Henter ut vektorene som peker fra leddet til begge nabopunktene
        u = points[0] - points[1]
        v = points[2] - points[1]

        # Finner vinkelen mellom vektorene, denne ligger mellom 0 og PI
        a = angleBetweenVectors(u, v)

        return math.pi - a

    def fingerCurvature(self, finger):
        return sum([self.jointCurvature(joint) for joint in getFingerJoints(finger)])

    def isFingerClosed(self, finger):
        return self.fingerCurvature(finger) > math.pi / 2

    def set_finger_states(self):
        self.finger_states = [self.isFingerClosed(finger) for finger in range(5)]
