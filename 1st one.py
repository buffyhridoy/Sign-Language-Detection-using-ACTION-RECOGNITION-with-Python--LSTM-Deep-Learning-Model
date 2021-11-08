#importing dependencies
import mediapipe as mp
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import os

#video capture from webcam using opencv
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('OpenCV Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

