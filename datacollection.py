import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Use macOS-compatible webcam capture
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Check if the camera is opened
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "/Users/saniyapunmagar/Desktop/Sign Language/Data/Hello"
os.makedirs(folder, exist_ok=True)  

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read frame.")
        break

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

        imgCropShape = imgCrop.shape
        if imgCrop.size == 0:
            continue  

        aspectratio = h / w

        if aspectratio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
