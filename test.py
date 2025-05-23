import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("keras_model.h5", "labels.txt")

offset = 20
imgSize = 300
confidence_threshold = 0.98  # Set minimum confidence threshold (80%)

labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        # Get image dimensions
        img_height, img_width = img.shape[:2]
        
        if len(hands) == 2:
            # Get bounding boxes for both hands
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']
            
            # Calculate combined bounding box
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
        else:
            # Single hand bounding box
            x, y, w, h = hands[0]['bbox']
        
        # Calculate safe coordinates for cropping
        crop_x1 = max(0, x - offset)
        crop_y1 = max(0, y - offset)
        crop_x2 = min(img_width, x + w + offset)
        crop_y2 = min(img_height, y + h + offset)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[crop_y1:crop_y2, crop_x1:crop_x2]

        if imgCrop.size != 0:  # Check if crop is valid
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

            # Get the highest confidence prediction
            max_confidence = max(prediction)
            
            # Draw bounding box
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                        (x + w+offset, y + h+offset), (255, 0, 255), 4)

            # Only show prediction if confidence is above threshold
            if max_confidence > confidence_threshold:
                # Ensure index is within valid range
                index = min(index, len(labels) - 1)
                
                # Draw prediction label with confidence
                cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                            (x - offset+170, y - offset-50+50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, f"{labels[index]} ({max_confidence:.1%})", 
                           (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Add q to quit
        break

cap.release()
cv2.destroyAllWindows()