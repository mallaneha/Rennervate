
import cv2

# creating an object, zero for external camera
cap = cv2.VideoCapture(0)

A = 0

if cap.isOpened():
    CHECK, frame = cap.read()
else:
    CHECK = False

# # for photo capture
# img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Image", img)
# cv2.waitKey(0)

cv2.namedWindow("Capturing")

# for video capture
while CHECK:
    A = A + 1

    # creating a frame object
    CHECK, frame = cap.read()
    print(CHECK)
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

print(A)

cv2.destroyAllWindows()

# shut down camera
cap.release()
