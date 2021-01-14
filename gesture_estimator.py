from cv2 import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import math
import numpy as np


gestures = {
    'OPEN' : [False] * 5,
    'CLOSED' : [True] * 5,
    'PEACE' : [True, False, False, True, True],
    'THUMBS_UP' : [False, True, True, True, True],
    'POINTING' : [True, False, True, True, True],
    'ROCK' : [True, False, True, True, False]
}

def getFingerJoints(finger):
    return range(4 * finger + 1, 4 * finger + 4)

def landmarkToVector(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])

class GestureEstimator:
    def __init__(self, landmarks: landmark_pb2.NormalizedLandmarkList):
        self.landmarks = landmarks.landmark
        self.set_finger_states()

    def getGesture(self):
        found = False
        for possibleGesture in gestures:
            if gestures[possibleGesture] == self.finger_states:
                gesture = possibleGesture
                found = True

        fallback = lambda closedFingers: ''.join([{False : 'O', True : 'C'}[f] for f in closedFingers])

        if found: return gesture
        else: return fallback(self.finger_states) 

    def jointCurvature(self, joint):
        landmarkTriplet = self.landmarks[joint - 1 : joint + 2].copy()

        if (joint - 1) % 4 == 0: landmarkTriplet[0] = self.landmarks[0]
         
        points = list(map(lambda p: np.array([p.x, p.y, p.z]), landmarkTriplet))

        u = points[0] - points[1]
        v = points[2] - points[1]

        magnitude = lambda v: math.sqrt(sum(map(lambda X: math.pow(X, 2), v)))
        angleBetweenVecs = lambda u, v: math.acos(np.divide(np.dot(u, v), magnitude(u) * magnitude(v)))
        a = angleBetweenVecs(u, v)

        return math.pi - a


    def fingerCurvature(self, finger):
        return sum([self.jointCurvature(joint) for joint in getFingerJoints(finger)])

    def isFingerClosed(self, finger):
        return self.fingerCurvature(finger) > math.pi / 2

    def set_finger_states(self):
        self.finger_states = [self.isFingerClosed(finger) for finger in range(5)]


