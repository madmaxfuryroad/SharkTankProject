import time
import cv2
import keras
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import tensorflow as tf
import numpy as np
import math


cap = cv2.VideoCapture(0) #access the camera
detector = HandDetector(maxHands=1)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', "Z"]
classifier = Classifier("keras_model.h5", "labels.txt")

padding = 30
imgSize = 300
counter = 0
COLOR_BOX_TUPLE = (90, 183, 204)



while True:
    success, img = cap.read() #reading every single pixel and return if it was successfull or not
    imgOutput = img.copy() #copy image to put the letters on
    hands, img = detector.findHands(img)
    #print(hands) #list containing a dictionary
    if hands: #if a list is populated . . .
        hand = hands[0] #take the first hand that is shown
        x,y,w,h = hand['bbox'] #actual box
        x1 = max(x-padding, 0)
        y1 = max(y-padding, 0)
        x2 = min(x + w + padding, img.shape[1])
        y2 = min(y + h + padding, img.shape[0])

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        #print(imgWhite)
        #making a square crop image - create a matrix of 1s



        if x2 > x1 and y2 > y1:
            imageCrop = img[y1:y2, x1:x2]
            if img.shape[1] > 0 and img.shape[0] > 0:
                #prevent imageCrop.shape[0] to go above the imgWhite == 300

                #height = min(imageCrop.shape[0], imgWhite.shape[0])
                #width = min(imageCrop.shape[1], imgWhite.shape[1])
                #imageCropResize = imageCrop[0:height, 0:width]

                aspectRatio = h/w
                if aspectRatio > 1:
                    k = imgSize/h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imageCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize-wCal)/2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize  # crop both image height and imgWhite
                    prediction, index = classifier.getPrediction(imgWhite) # PREDICTION
                    #print(prediction)


                else:
                    k = imgSize/w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imageCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize-hCal)/2)
                    imgWhite[hGap:hCal+hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite) # PREDICTION
                    #print(prediction)

                cv2.rectangle(imgOutput, (x-padding, y-padding), (x+w+padding, y+h+padding), (2,79,234), 4) #rectangle box
                cv2.rectangle(imgOutput, (x-padding, y-padding-80), (x+w+padding-50, y-padding), (2,79,234), cv2.FILLED) # letter box
                cv2.putText(imgOutput, labels[index], (x,y-50), cv2.FONT_HERSHEY_DUPLEX,2,(0,0,0),2)


                #print(labels[index])
                #cv2.rectangle(imgOutput, (x+w,y+h), COLOR_BOX_TUPLE, 4)

                #cv2.imshow("imageCrop", imageCrop)
                #cv2.imshow("imgWhite", imgWhite)


    else:
        cv2.destroyWindow("imageCrop")
        cv2.destroyWindow("imgWhite")

    cv2.imshow("Sign Language Detector", imgOutput) #img pixels displayed in a window called "Image"
    key = cv2.waitKey(1)