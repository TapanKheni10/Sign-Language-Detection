import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import time
import math
import uuid
 
# PARAMETERS AND VARIABLES
height = 480
width = 640
offset = 20
imgSize = 300
class_idx = 0
number_of_images = 100

class_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "3", "4"]

def create_processed_image(frame, hands):
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

        else:
            factor = imgSize / w
            newHeight = math.ceil(h * factor)

            imgResized = cv2.resize(imgCrop, (imgSize, newHeight))
            heightGap = math.ceil((imgSize - newHeight) / 2)

            imgBlank[heightGap : newHeight + heightGap, :] = imgResized

        return imgBlank, imgCrop

    return None, None

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = HandDetector(maxHands=1)

while True:

    ret, frame = cap.read()

    key = cv2.waitKey(1)

    if key == ord('s'):
        class_name = class_list[class_idx]

        print(f"Collecting images for class: {class_name}")

        directory_path = f'database/{class_name}'

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)


        for img_num in range(number_of_images):

            _, frame = cap.read()
            hands, frame = detector.findHands(frame)

            try:
                if hands:
                    imgBlank, imgCrop = create_processed_image(frame, hands)
                    cv2.imwrite(f"{directory_path}/{img_num + 1}_{str(uuid.uuid1())}.jpg", imgBlank)
                    print(f"Image {img_num + 1} is saved")
                    cv2.imshow("Saved Image", imgBlank)
                    cv2.waitKey(1000)
            except:
                print("No hand detected")

            cv2.imshow("Image", frame)

            time.sleep(2)
        
        print(f"Collection of images is completed for class: {class_name}")
        class_idx += 1
        
        if class_idx < len(class_list):
            print("Press 's' to collect images for the next class.")

        else:
            print("Images for all classes are collected. Press 'q' to exit.")
    
    cv2.imshow("Image", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Data collection is completed")