
from typing import OrderedDict
import argparse
import cv2
import dlib


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on systesm")
args = vars(ap.parse_args())

# P = "shape_predictor_68_face_landmarks.dat"
print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# For dlib's 68-point facial detector:
FACIAL_LANDMARKS_INDEX = OrderedDict([
    ("jaw", (0, 17)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("nose", (27, 36)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("mouth", (48, 68))
])


def main():
    # using 0 for external camera input
    cap = cv2.VideoCapture(args["webcam"])

    if cap.isOpened():
        CHECK, frame = cap.read()
    else:
        CHECK = False

    while CHECK:
        CHECK, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for (i, face) in enumerate(faces):
            x1 = face.left()
            x2 = face.right()
            y1 = face.top()
            y2 = face.bottom()

            # draw the face bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # show the face number
            cv2.putText(frame, "Face #{}".format(i + 1), (x1-10, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            landmarks = predictor(gray, face)

            # draw the facial landmamrks
            for n in range(36, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        cv2.namedWindow("Capturing")

        cv2.imshow("Capturing", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            print("Ending the capture")
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
