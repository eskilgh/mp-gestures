from cv2 import cv2
import mediapipe as mp
import numpy as np
import math
from util import (
        draw_handmarks_label,
        draw_landmark_bbox
 )
from gesture_estimator import GestureEstimator

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
        min_detection_confidence = 0.8, 
        min_tracking_confidence = 0.5,
        max_num_hands = 2
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            draw_landmark_bbox(image, hand_landmarks)

            gestEst = GestureEstimator(hand_landmarks)
            gesture = gestEst.getGesture()

            draw_handmarks_label(image, gesture, hand_landmarks) 
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
           
    cv2.imshow('Angles of joints', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
