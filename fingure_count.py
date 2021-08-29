import numpy as np
import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import os
wCam,hCam = 720,1280
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
folderPath = "C:/Users/Balaji/Documents/Machine Learning/AI Virtual Mouse/finger"
myList = os.listdir(folderPath)
print(myList)
detector = htm.handDetector(detectionCon=0.2)
overlayList =[]
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))    
tipid = [4,8,12,16,20]
while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findposition(img,draw=False)
    #print(lmList)
    if len(lmList) != 0:
        finger =[]
        #thumb
        if lmList[tipid[0]][1] > lmList[tipid[0]-1][1]:
            finger.append(1)
        else:
            finger.append(0)
        #other 4 fingures    
        for id in range(1,5):
            if lmList[tipid[id]][2] < lmList[tipid[id]-2][2]:
                finger.append(1)
            else:
                finger.append(0)
        totalfingures = finger.count(1)
        print(totalfingures)        
        #print(finger)            
        h,w,c = overlayList[totalfingures-1].shape
        img[0:h,0:w] = overlayList[totalfingures-1]
        cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalfingures),(45,375),cv2.FONT_HERSHEY_PLAIN, 10,(255,0,0),25)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)