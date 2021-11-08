#importing dependencies
import mediapipe as mp
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import os

mp_holistic = mp.solutions.holistic  #holistic model
mp_drawing = mp.solutions.drawing_utils #drawing utils

#mediapipe detection function
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results, image

#dran landmark function
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.Face_Connections)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.Hand_Connections)  
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.Hand_Connections)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_Connections)
    


#using holistic model in opencv feed
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
    while cap.isOpened():
        ret, frame = cap.read()
        results, image = mediapipe_detection(frame, model)
        print(results)
        #mp_drawing.draw_detection_results(image, results)
        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

print(results.face_landmarks)
print(results.face_landmarks.landmark)
print(len(results.face_landmarks.landmark))
print(len(results.pose_landmarks.landmark))
print(len(results.left_hand_landmarks.landmark))