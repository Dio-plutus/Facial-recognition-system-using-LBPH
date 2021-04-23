## Import necessary packages
import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

# Load HAAR face classifier
face_cascade_path = "assets/face_detection/haarcascade_frontalface_default.xml"
left_eye_cascade_path = "assets/face_detection/haarcascade_left_eye.xml"
right_eye_cascade_path = "assets/face_detection/haarcascade_right_eye.xml"
face_cascade = cv.CascadeClassifier(face_cascade_path)
left_eye_cascade = cv.CascadeClassifier(left_eye_cascade_path)
right_eye_cascade = cv.CascadeClassifier(right_eye_cascade_path)

# defines code for face detector
def face_detector(gray_img):
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    if faces is ():
        return None, (None, None, None,None)
    
    for (x,y,w,h) in faces:
        cv.rectangle(gray_img, (x,y), (x+w, y+h), (0, 255, 255), 2)
        cropped_face = gray_img[y:y+h, x:x+w]
        roi = cv.resize(cropped_face, (200, 200))
        coord = (x,y,w,h)

    return roi, coord


def preprocess_image(gray_img):
    height, width = gray_img.shape[:2]

    midX = int(width/2)
    leftSide = gray_img[int(0):height, int(0):midX]
    rightSide = gray_img[int(0):height, midX:width]

    equL = cv.equalizeHist(leftSide)
    equR = cv.equalizeHist(rightSide)

    img_rotated_equalized = np.concatenate((equL, equR), axis=1)
    img = cv.bilateralFilter(img_rotated_equalized, 15, 75, 75)
    return img
