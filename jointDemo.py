from cv2 import cv2
import mediapipe as mp
import numpy as np
import math

# Tar inn listen over ledd og indeksen til et ledd hvor vi ønsker å finne vinkelen
# Merk at her må n være et punkt mellom 4f + 1 til og med 4f + 3
# Dette er fordi vi kan kun måle vinkel i et nøkkelpunkt dersom det er et ledd
def jointCurvature(lm, n):
    # Henter koordinatene til forrige, nåværende og neste ledd
    points = lm[n - 1 : n + 2]

    # Dersom punktet vi står i er det første leddet på en finger
    # setter vi det forrige leddet til håndleddet
    if (n - 1) % 4 == 0:
        points[0] = lm[0]

    # Trekker fra midtpunktet fra begge endepunktene for å få vektorene vi ønsker
    u = points[0] - points[1]
    v = points[2] - points[1]

    # Finner vinkelen mellom vektorene, denne ligger mellom 0 og PI
    a = angleBetweenVecs(u, v)

    # Regner ut et tall for krumningen som vil variere mellom 0 og 1/2 PI
    # Tilsvarende helt rett til helt krum
    curvature = math.pi - a

    return curvature

# Inversen av dotproduktet for å finne vinkelen mellom to vektorer 
angleBetweenVecs = lambda u, v: math.acos(np.divide(np.dot(u, v), magnitude(u) * magnitude(v)))

# Magnituden til en vilkårlig dimensjonal vektor v
# Merk at vektoren v må være itererbar
magnitude = lambda v: math.sqrt(sum(map(lambda X: math.pow(X, 2), v)))

# Tar inn listen over alle nøkkelpunkter og indeksen til en finger
# Returnerer summen av alle vinklene på den fingeren
fingerCurvature = lambda lm, f: sum([jointCurvature(lm, n) for n in range(4 * f + 1, 4 * f + 3)])

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
        min_detection_confidence = 0.5, 
        min_tracking_confidence = 0.5,
        max_num_hands = 1
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
            
            # Hente hele lista med punkter
            lm = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark])
            
            # Escape symbol for clearing av terminal
            print(chr(27) + "[2J")

            # Terminal output med fingre
            for finger in range(5):
                print('Finger: {}, Curvature {}'.format(finger, fingerCurvature(lm, finger)))
           
            # Dirty newline
            print('\n')

            # Terminal output med ledd
            for finger in range(5):
                for joint in range(4 * finger + 1, 4 * finger + 3):
                    print('Joint: {}, Curvature {}'.format(joint, jointCurvature(lm, joint)))

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Angles of joints', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
