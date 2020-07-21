
from typing import OrderedDict
import cv2
import dlib

cap = cv2.VideoCapture(0)

if cap.isOpened():
    CHECK, frame = cap.read()
else:
    CHECK = False

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

FACIAL_LANDMARKS_INDEX = OrderedDict([
    ("jaw", (0, 17)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("nose", (27, 36)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("mouth", (48, 64))
])

while CHECK:
    CHECK, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(gray, face)

        for n in range(36, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.namedWindow("Capturing")

    cv2.imshow("Capturing", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
