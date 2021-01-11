from cv2 import cv2
import mediapipe as mp
from gesture_classifier import GestureClassifier, draw_label, draw_landmark_bbox
from gesture_calc import GestureCalculator

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:

hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2
)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_landmark_bbox(image, hand_landmarks)
            gesture_calc = GestureCalculator(hand_landmarks.landmark)
            gest_code = gesture_calc.process()
            if gest_code:
                draw_label(image, gest_code, (100, 100), (255, 0, 0))
        # gc = GestureClassifier(results.multi_hand_landmarks)
        # gesture_code = gc.process()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cap.release()
