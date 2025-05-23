#optimized code but doesn't send entry to thingspeak.
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import aiohttp
import asyncio
 # ThingSpeak settings
THINGSPEAK_API_KEY = "4J7NVXBAIWG0O7MN"
THINGSPEAK_UPDATE_URL = "https://api.thingspeak.com/update?api_key=" + THINGSPEAK_API_KEY
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:\PycharmProjects\pythonProject1\Model/keras_model.h5", "C:\PycharmProjects\pythonProject1\Model/labels.txt")
offset = 20
imgSize = 300
labels = ["A", "B","C"]
async def send_to_thingspeak(predicted_label):
    """Send predicted label to ThingSpeak using an asynchronous HTTP request."""
    async with aiohttp.ClientSession() as session:
        url = f"{THINGSPEAK_UPDATE_URL}&field1={predicted_label}"
        await session.get(url)
while True:
    success, img = cap.read()
    if not success:
        continue
    imgOutput = img.copy()
    hands, _ = detector.findHands(img)
    if not hands:
        cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)
        continue
    hand = hands[0]
    x, y, w, h = hand['bbox']
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
    imgCropShape = imgCrop.shape
    if imgCropShape[0] > 0 and imgCropShape[1] > 0:
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w+offset, y + h+offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
        predicted_index = classifier.getPrediction(imgWhite, draw=False)[1]
        predicted_label = labels[predicted_index]
        if predicted_label == "B":
            # Turn on the light
            asyncio.ensure_future(send_to_thingspeak("1"))
        elif predicted_label == "A":
            # Turn off the light
            asyncio.ensure_future(send_to_thingspeak("0"))
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()