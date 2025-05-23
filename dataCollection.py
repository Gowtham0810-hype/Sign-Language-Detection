import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder = "data-images/C"
if not os.path.exists(folder):
    os.makedirs(folder)

counter = 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        img_height, img_width = img.shape[:2]
        
        if len(hands) == 2:
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']
            
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
        else:
            x, y, w, h = hands[0]['bbox']
        
        crop_x1 = max(0, x - offset)
        crop_y1 = max(0, y - offset)
        crop_x2 = min(img_width, x + w + offset)
        crop_y2 = min(img_height, y + h + offset)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[crop_y1:crop_y2, crop_x1:crop_x2]

        if imgCrop.size != 0:
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

            cv2.rectangle(imgOutput, (x-offset, y-offset),
                        (x + w+offset, y + h+offset), (255, 0, 255), 4)
            
            #hands_text = f"Hands: {len(hands)}"
            #cv2.putText(imgOutput, hands_text, (10, 30), 
                       #cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

          #  cv2.imshow("ImageCrop", imgCrop)
           # cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    
    if key == ord("s"):
        if len(hands) == 2:
            save_folder = os.path.join(folder, "dual")
        else:
            save_folder = os.path.join(folder, "single")
            
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        counter += 1
        cv2.imwrite(f'{save_folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter} to {save_folder}")
    
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
