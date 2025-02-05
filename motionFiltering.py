import cv2 as cv

video = cv.VideoCapture(0)
subtractor = cv.createBackgroundSubtractorMOG2(300, 50)

while True:
    ret, frame = video.read()

    if ret:
        mask = subtractor.apply(frame)
        cv.imshow('Mask', mask)

        if cv.waitKey(5) == ord('X'):
            break
cv.destroyAllWindows()
video.release()
