# importing the required libraries
import os
from typing import OrderedDict
import time
import datetime
import threading
import csv
from pydub import AudioSegment
from pydub.playback import play
import requests
from tqdm import tqdm
import numpy as np
import cv2
import dlib

start_time = time.time()


def download_detector():
    "downloading the required dlib 68 landmarks predictor"
    url = "https://github.com/JeffTrain/selfie/raw/master/shape_predictor_68_face_landmarks.dat"
    local_filename = url.split("/")[-1]

    if not os.path.exists(local_filename):
        response = requests.get(url, stream=True)
        length = response.headers.get("content-length", 0)
        print("Downloading the shape predictor file:")
        with tqdm.wrapattr(
            open(local_filename, "wb"),
            "write",
            miniters=1,
            desc="Downloading file ",
            total=int(length),
        ) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)
        print("Download complete!")
        time.sleep(0.1)


def midpoint(p1, p2):
    "for calculating the midpoint between two points"
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def euclidean_distance(leftx, lefty, rightx, righty):
    "for calculating the distance between two points"
    return np.sqrt((leftx - rightx) ** 2 + (lefty - righty) ** 2)


def eye_aspect_ratio(eye_point, facial_landmark):
    "for calculating the eye aspect ratio"
    left_point = [
        facial_landmark.part(eye_point[0]).x,
        facial_landmark.part(eye_point[0]).y,
    ]
    right_point = [
        facial_landmark.part(eye_point[3]).x,
        facial_landmark.part(eye_point[3]).y,
    ]

    top_point = midpoint(
        facial_landmark.part(eye_point[1]), facial_landmark.part(eye_point[2])
    )
    bottom_point = midpoint(
        facial_landmark.part(eye_point[4]), facial_landmark.part(eye_point[5])
    )

    horizontal_dist = euclidean_distance(
        left_point[0], left_point[1], right_point[0], right_point[1]
    )
    vertical_dist = euclidean_distance(
        top_point[0], top_point[1], bottom_point[0], bottom_point[1]
    )

    EAR = vertical_dist / horizontal_dist
    return EAR


def mouth_aspect_ratio(mouth_point, landmark):
    "for calculating mouth aspect ratio"
    # calculating distance of the horizontal line
    left_horizontal = [landmark.part(mouth_point[0]).x, landmark.part(mouth_point[0]).y]
    right_horizontal = [
        landmark.part(mouth_point[6]).x,
        landmark.part(mouth_point[6]).y,
    ]
    horizontal_dist = euclidean_distance(
        left_horizontal[0], left_horizontal[1], right_horizontal[0], right_horizontal[1]
    )

    # calculating distance of left vertical line
    top_left_vertical = [
        landmark.part(mouth_point[2]).x,
        landmark.part(mouth_point[2]).y,
    ]
    bot_left_vertical = [
        landmark.part(mouth_point[10]).x,
        landmark.part(mouth_point[10]).y,
    ]
    left_vertcal_dist = euclidean_distance(
        top_left_vertical[0],
        top_left_vertical[1],
        bot_left_vertical[0],
        bot_left_vertical[1],
    )

    # calculating distance of mid vertical line
    top_mid_vertical = [
        landmark.part(mouth_point[3]).x,
        landmark.part(mouth_point[3]).y,
    ]
    bot_mid_vertical = [
        landmark.part(mouth_point[9]).x,
        landmark.part(mouth_point[9]).y,
    ]
    mid_vertical_dist = euclidean_distance(
        top_mid_vertical[0],
        top_mid_vertical[1],
        bot_mid_vertical[0],
        bot_mid_vertical[1],
    )

    # calculating distance of right vertical line
    top_right_vertical = [
        landmark.part(mouth_point[4]).x,
        landmark.part(mouth_point[4]).y,
    ]
    bot_right_vertical = [
        landmark.part(mouth_point[8]).x,
        landmark.part(mouth_point[8]).y,
    ]
    right_vertical_dist = euclidean_distance(
        top_right_vertical[0],
        top_right_vertical[1],
        bot_right_vertical[0],
        bot_right_vertical[1],
    )

    MAR = (left_vertcal_dist + mid_vertical_dist + right_vertical_dist) / (
        3 * horizontal_dist
    )
    return MAR


def raise_alarm():
    "used to play the alarm sound on loop"
    alert_sound = AudioSegment.from_wav("audio/beep-06.wav")
    while ALARM_ON:
        play(alert_sound)


def logger(message):
    "used to log messages"
    if __debug__:
        print(message)


def save_ear(ear_list, mar_list, filename):
    # if not os.path.exists(f"{filename}.csv"):
    #     with open(f"{filename}.csv", mode="w") as train_file:
    #         file_write = csv.writer(
    #             train_file, delimiter=",", quoting=csv.QUOTE_MINIMAL
    #         )
    #         file_write.writerow(ear_list)
    #         file_write.writerow(mar_list)
    # else:
    with open(f"{filename}.csv", mode="a") as file:
        file_write = csv.writer(file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        file_write.writerow(ear_list)
        file_write.writerow(mar_list)


def main():
    EAR_THRESH = 0.25
    EAR_CONSECUTIVE_FRAMES = 42

    COUNTER = 0
    count = 0
    global ALARM_ON
    # ALARM_ON = False

    print("Preparing the detectors:")
    download_detector()
    print("Loading modules:")

    start = datetime.datetime.now()
    P = "shape_predictor_68_face_landmarks.dat"
    print("Loading facial landmark predictor...")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(P)
    after_model_load = datetime.datetime.now()

    # For dlib's 68-point facial detector:
    FACIAL_LANDMARKS = OrderedDict(
        [
            ("jaw", list(range(0, 17))),
            ("right_eyebrow", list(range(17, 22))),
            ("left_eyebrow", list(range(22, 27))),
            ("nose", list(range(27, 36))),
            ("right_eye", list(range(36, 42))),
            ("left_eye", list(range(42, 48))),
            ("mouth", list(range(48, 68))),
        ]
    )

    basepath = "videos/"
    # for file in os.listdir(basepath):
    #     if os.path.isdir(os.path.join(basepath, file)):
    #         for f in os.listdir(os.path.join(basepath,file)):
    #             print(f)

    for filename in os.listdir(basepath):
        # using 0 for external camera input
        # cap = cv2.VideoCapture(0)
        if os.path.isfile(os.path.join(basepath, filename)):
            cap = cv2.VideoCapture(os.path.join(basepath, filename))

            if cap.isOpened():
                CHECK, frame = cap.read()
            else:
                CHECK = False

            time_stamp = True

            ear_list = []
            mar_list = []

            ear_start_time = time.time()
            while CHECK:
                _, frame = cap.read()

                if _:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    before_face = datetime.datetime.now()
                    faces = detector(gray)
                    after_face = datetime.datetime.now()

                    for (i, face) in enumerate(faces):
                        x1 = face.left()
                        x2 = face.right()
                        y1 = face.top()
                        y2 = face.bottom()

                        # draw the face bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # show the face number
                        cv2.putText(
                            frame,
                            "Face #{}".format(i + 1),
                            (x1 - 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                        before_landmarks = time.time()
                        landmarks = predictor(gray, face)
                        after_landmarks = time.time()

                        # calculating the facial landmamrks
                        landmark_keys = ["right_eye", "left_eye", "mouth"]
                        required_landmarks = []
                        for key in landmark_keys:
                            required_landmarks.extend(FACIAL_LANDMARKS.get(key))

                        # drawing the facial landmarks in the video
                        for n in required_landmarks:
                            x = landmarks.part(n).x
                            y = landmarks.part(n).y
                            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

                        calculated_mar = mouth_aspect_ratio(
                            FACIAL_LANDMARKS["mouth"], landmarks
                        )

                        left_EAR = eye_aspect_ratio(
                            FACIAL_LANDMARKS["left_eye"], landmarks
                        )
                        right_EAR = eye_aspect_ratio(
                            FACIAL_LANDMARKS["right_eye"], landmarks
                        )

                        ear_both_eyes = (left_EAR + right_EAR) / 2

                        # count += 1

                        if (time.time() - ear_start_time) >= 4:
                            ear_list.append(round(ear_both_eyes, 2))
                            mar_list.append(round(calculated_mar, 2))
                            # print("4 sec")
                            ear_start_time = time.time()

                            # ear_time = time.time()
                            # count = 0

                        if ear_both_eyes < EAR_THRESH:
                            COUNTER += 1

                            if COUNTER >= EAR_CONSECUTIVE_FRAMES:
                                if not ALARM_ON:
                                    ALARM_ON = True

                                    # creating new thread to play the alarm in background
                                    audio_thread = threading.Thread(target=raise_alarm)
                                    audio_thread.start()

                                cv2.putText(
                                    frame,
                                    "Drowsiness Alert!",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2,
                                )
                                # print("Drowsiness detected!")
                        else:
                            COUNTER = 0
                            ALARM_ON = False

                        cv2.putText(
                            frame,
                            "MAR: {:.2f}".format(calculated_mar),
                            (300, 400),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )

                        cv2.putText(
                            frame,
                            "EAR: {:.2f}".format(ear_both_eyes),
                            (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )
                else:
                    print("End of video")
                    break
                cv2.namedWindow("Capturing")
                cv2.imshow("Capturing", frame)

                if time_stamp:
                    logger(
                        "---{} seconds---".format(round((time.time() - start_time), 2))
                    )
                    logger("Model load: " + str(after_model_load - start))
                    logger("Face detection: " + str(after_face - before_face))
                    if len(faces) > 0:
                        logger(
                            "Landmark detection: "
                            + str(after_landmarks - before_landmarks)
                        )
                    time_stamp = False

                key = cv2.waitKey(1)

                # Use q to close the detection
                if key == ord("q"):
                    print("Ending the capture")
                    break

            # print(ear_list)
            save_ear(ear_list, mar_list, "check")

            cv2.destroyAllWindows()
            cap.release()


if __name__ == "__main__":
    main()
