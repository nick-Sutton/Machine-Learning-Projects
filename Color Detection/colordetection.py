import cv2, numpy
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_purple = numpy.array([90, 50, 50])
    upper_purple = numpy.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    results = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', results)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()