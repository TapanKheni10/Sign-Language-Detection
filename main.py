import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from cvzone.ClassificationModule import Classifier
import math

# PARAMETERS AND VARIABLES
offset = 20
imgSize = 300

class_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "3", "4"]

def get_sign_prediction(frame, hands, img):
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = frame[y - offset : y + h + offset, x - offset : x + w + offset]

        # creating a blank image
        imgBlank = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w

        if aspectRatio > 1:
            factor = imgSize / h
            newWidth = math.ceil(w * factor)

            imgResized = cv2.resize(imgCrop, (newWidth, imgSize))
            widthGap = math.ceil((imgSize - newWidth) / 2)

            imgBlank[:, widthGap : newWidth + widthGap] = imgResized

            prediction, class_idx = classifier.getPrediction(imgBlank, draw = False)

        else:
            factor = imgSize / w
            newHeight = math.ceil(h * factor)

            imgResized = cv2.resize(imgCrop, (imgSize, newHeight))
            heightGap = math.ceil((imgSize - newHeight) / 2)

            imgBlank[heightGap : newHeight + heightGap, :] = imgResized

            prediction, class_idx = classifier.getPrediction(imgBlank, draw = False)

        cv2.rectangle(imgCopy, (x - offset - 20, y - offset - 20), (x + w + offset, y + h + offset), (255, 0, 255), 4)
        cv2.rectangle(imgCopy, (x - offset - 20, y - offset - 90), (x + w + offset, y - offset - 90 + 70), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgCopy, class_list[class_idx], (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.imshow("Image", img),
        print(prediction, class_list[class_idx])

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

while True:

    ret, frame = cap.read()
    imgCopy = frame.copy()
    hands, frame = detector.findHands(frame)
    get_sign_prediction(frame, hands, imgCopy)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
