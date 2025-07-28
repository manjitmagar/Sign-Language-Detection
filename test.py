import os
import time
import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Paths
tflite_model_path = "/Users/saniyapunmagar/Desktop/Sign Language/converted_keras/converted_model.tflite"
labels_path = "/Users/saniyapunmagar/Desktop/Sign Language/converted_keras/labels.txt"
save_folder = "/Users/saniyapunmagar/Desktop/Sign Language/Data/Hello"

# Initialize camera and modules
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  
detector = HandDetector(maxHands=1)
classifier = Classifier(modelPath="/Users/saniyapunmagar/Desktop/converted_keras/keras_model.h5",
                        labelsPath="/Users/saniyapunmagar/Desktop/converted_keras/labels.txt")

offset = 20
imgSize = 300
counter = 0

# Load labels
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
else:
    labels = ["Hello", "Thank you"]

os.makedirs(save_folder, exist_ok=True)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        
        text = labels[index]
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 2
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        rect_x1 = x - offset
        rect_y1 = y - offset - text_height - 20
        rect_x2 = rect_x1 + text_width + 20
        rect_y2 = y - offset

        rect_x1 = max(0, rect_x1)
        rect_y1 = max(0, rect_y1)
        rect_x2 = min(img.shape[1], rect_x2)

        cv2.rectangle(imgOutput, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, text, (rect_x1 + 10, rect_y2 - 10), font, font_scale, (0, 0, 0), thickness)

        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    key = cv2.waitKey(1)

    if key == ord('s'):
        counter += 1
        filename = f'{save_folder}/Image_{int(time.time())}.jpg'
        cv2.imwrite(filename, imgWhite)
        print(f"Saved {filename} - Count: {counter}")
    elif key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
