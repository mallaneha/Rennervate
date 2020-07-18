import cv2
import time

# creating an opject, zero for external camera
cap = cv2.VideoCapture(0)

a = 0

if cap.isOpened():
    check, frame = cap.read()
else:
    check = False

# # for photo capture
# img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Image", img)
# cv2.waitKey(0)

cv2.namedWindow("Capturing")

# for video capture
while check:
    a = a + 1

    # creating a frame object
    check, frame = cap.read()
    print(check)
    print(frame)

    # converting to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # show the frame
    cv2.imshow("Capturing", gray)
    # cv2.imshow("Normal", frame)

    # for playing
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

print(a)

cv2.destroyAllWindows()

# shut down camera
cap.release()
