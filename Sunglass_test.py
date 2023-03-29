import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from time import time
from solutions import image_resize
from solutions import checkRotation, correct_rotation, save_Crop, clean_Folder
from solutions import get_Aspect_Ratios, get_Crops_L_Eye, get_Minute
import os
import matplotlib.pyplot as plt
from PIL import Image



# If false it wont save any data-points into a csv
DEBUG = True
SAVE = False
PREDICT = False
PERSON = "Data_collection"

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
frame_count=0
EAR_LIST = []
frame_no = list()

first_image=False
second_image=False
def main() -> None:
    INPUT = input("""
    Enter the choice:
    1. Webcam
    2. Video Path
    """)
    path = f"./data/{PERSON}/"
    videofilename = []
    for x in os.listdir(path):
        if x.endswith(".mp4"):
            videofilename.append(x)
    #print(videofilename)
    videofilename.sort()
    for x in videofilename:
        if DEBUG:
            NAME = x.split('.')[0]

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        FRAME_COUNT = 0
        rotateCode = None
        length = 1800
        prev_frame_time, new_frame_time = 0, 0
        COUNTER, TOTAL = 0, 0
        frame_count=0
        EAR_LIST = []
        frame_no = list()

        # For webcam input:
        # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        if INPUT == "1":
            cap = cv2.VideoCapture(0)

        else:
            video = x
            print(x)
            path = f"./data/{PERSON}/{video}"
            print(path)
            cap = cv2.VideoCapture(path)

            rotateCode = checkRotation(path)
            #rotateCode = False

            length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
            print(f"LENGTH == {length}")

        # cap.set(3, WIDTH)
        # cap.set(4, HEIGHT)

        # Clean the folder and create a new file
        if DEBUG:
            clean_Folder(f"./Feature_Extraction/{PERSON}/{NAME}_EAR.csv", f"{NAME}_EAR.csv")

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():
                
                success, image = cap.read()
                if success:
                    WIDTH, HEIGHT = image.shape[0], image.shape[1]
                DATA = []

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # calculate the FPS
                new_frame_time = time()

                FPS = str(int(1 / (new_frame_time - prev_frame_time)))
                prev_frame_time = new_frame_time
                frame_count+=1
                # fix the image orientation
                if rotateCode:
                    image = correct_rotation(image, rotateCode)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                frame_count+=1
                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    person1_results = results.multi_face_landmarks[0].landmark
                    EAR_L, EAR_R,EAR = get_Aspect_Ratios(person1_results, "EAR")
                    print(EAR)
                    
# Visualize the Left and Region by drawing a rectangle on it on the actual image.
# RIGH EYE
                    rightEyeImg = getRightEye(image, person1_results)
                    rightEyeHeight, rightEyeWidth, _ = rightEyeImg.shape

                    xRightEye, yRightEye, rightEyeWidth, rightEyeHeight= getRightEyeRect(image, person1_results)
                    cv2.rectangle(image, (xRightEye, yRightEye),(xRightEye + rightEyeWidth, yRightEye + rightEyeHeight), (200, 21, 36), 2)

# LEFT EYE
                    leftEyeImg = getLeftEye(image, person1_results)
                    leftEyeHeight, leftEyeWidth, _ = leftEyeImg.shape
                    xLeftEye, yLeftEye, leftEyeWidth, leftEyeHeight= getLeftEyeRect(image, person1_results)
                    cv2.rectangle(image, (xLeftEye, yLeftEye),(xLeftEye + leftEyeWidth, yLeftEye + leftEyeHeight), (200, 21, 36), 2)
                    img1 = cv2.imread("testimag3.jpg",0)
                    img2 = cv2.imread("testimag4.jpg",0)
                    histg = cv2.calcHist([img1],[0],None,[256],[0,256]) 
                    histg2 = cv2.calcHist([img2],[0],None,[256],[0,256]) 
                    # print(histg)
                    # print(histg2)
                    cv2.imshow('MediaPipe Face Mesh', image)
                if cv2.waitKey(5) & 0xFF == ord("q") or FRAME_COUNT >= length:
                    break

        cap.release()

 #Crop the right eye region
def getRightEye(image, landmarks):
    eye_top = int(landmarks[386].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[374].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])
    right_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return right_eye

# Get the right eye coordinates on the actual -> to visualize the bbox
def getRightEyeRect(image, landmarks):
    global first_image
    eye_top = int(landmarks[386].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[374].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])

    cloned_image = image.copy()
    cropped_right_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    if(first_image==False):
        cv2.imwrite('./testimag11.jpg',cropped_right_eye)
        first_image=True
    h, w, _ = cropped_right_eye.shape
    x = eye_left
    y = eye_top
    return x, y, w, h


def getLeftEye(image, landmarks):
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[35].x * image.shape[1])
    eye_bottom = int(landmarks[153].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])
    left_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return left_eye


def getLeftEyeRect(image, landmarks):
    # eye_left landmarks (27, 23, 130, 133) ->? how to utilize z info
    global second_image
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[33].x * image.shape[1])
    eye_bottom = int(landmarks[145].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])

    cloned_image = image.copy()
    cropped_left_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    if(second_image==False):
        cv2.imwrite('./testimag12.jpg',cropped_left_eye)
        second_image=True
    h, w, _ = cropped_left_eye.shape

    x = eye_left
    y = eye_top
    return x, y, w, h

if __name__ == "__main__":
    main()
