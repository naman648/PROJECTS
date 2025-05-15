import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time

# Eye Aspect Ratio calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # vertical line 1
    B = dist.euclidean(eye[2], eye[4])  # vertical line 2
    C = dist.euclidean(eye[0], eye[3])  # horizontal line
    ear = (A + B) / (2.0 * C)
    return ear

# Constants
EYE_AR_THRESH = 0.25  # below this is considered closed
EYE_AR_CONSEC_FRAMES = 20  # if closed for this many frames = drowsy

# Counters
COUNTER = 0
TOTAL_BLINKS = 0
DROWSY_ALERT = False

# Initialize Dlib's face detector and landmark predictor
print("[INFO] Loading models...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\91626\Neuro_Vision\shape_predictor_68_face_landmarks.dat")

# Grab landmark indexes for eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start video stream
cap = cv2.VideoCapture(0)
print("[INFO] Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eyes
        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

        # Blink logic
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                DROWSY_ALERT = True
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if COUNTER >= 3:
                TOTAL_BLINKS += 1
            COUNTER = 0
            DROWSY_ALERT = False

        # Display stats
        cv2.putText(frame, f"Blinks: {TOTAL_BLINKS}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Fatigue Monitor", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
