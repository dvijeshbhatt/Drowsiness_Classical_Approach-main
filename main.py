import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from time import time
from solutions import image_resize
from solutions import checkRotation, correct_rotation, save_Crop, clean_Folder
from solutions import get_Aspect_Ratios, get_Crops_L_Eye, get_Minute
import os

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
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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
                    # annotations
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())

                    person1_results = results.multi_face_landmarks[0].landmark

                    # Resize the image if greater than screen size
                    if image.shape[0] > 1270:
                        image = image_resize(image, height=640)
                        WIDTH, HEIGHT = image.shape[0], image.shape[1]

                    # FOR EAR (EYE ASPECT RATIO)
                    EAR_L, EAR_R,EAR = get_Aspect_Ratios(person1_results, "EAR")
                    EAR_LIST.append(EAR)
                    EAR_diff_LAR = abs(EAR_L-EAR_R)
                    newEAR=0
                    if EAR_diff_LAR<=0.35:
                        newEAR = EAR
                    else:
                        newEAR = min(EAR_L,EAR_R)
                    
                    MAR = get_Aspect_Ratios(person1_results, "MAR")
                    MOEAR = MAR / EAR

                    ''' if frame_count > 6:
                        if  EAR <= ((EAR_LIST[frame_count-5])*0.70) and EAR_diff_LAR<0.35: 
                            if frame_count - frameno !=1:
                                frameno = frame_count
                                frame_no.append(frameno)
                                COUNTER+=1
                            else:
                                frameno = frame_count
                            if COUNTER >=3:
                                blink_count+=1
                                COUNTER=0
                    '''
                    # ! get the crop points for left eye and show them/ save them
                    lt, w, h = get_Crops_L_Eye(person1_results, WIDTH, HEIGHT)

                    if SAVE and EAR > 0.15 and FRAME_COUNT % 20 == 0:
                        save_Crop(image, lt, w, h,
                                path=f"./data/eye/open/{FRAME_COUNT}_{NAME}.jpg")
                    elif SAVE and EAR < 0.08:
                        print(f"EAR = {EAR}")
                        save_Crop(image, lt, w, h,
                                path=f"./data/eye/closed/{FRAME_COUNT}_{NAME}.jpg")


                    # # standerdize the EAR
                    # if len(EAR_LIST) < 300:
                    #     EAR_LIST.append(EAR)
                
                    # else:
                    #     mean = np.mean(EAR_LIST)
                    #     std_dev = statistics.pstdev(EAR_LIST)

                    #     EAR = (EAR - mean) / std_dev

                    # put the EAR & MAR & MOEAR on the image
                    # Flip the image horizontally for a selfie-view display.
                    image = cv2.flip(image, 1)

                    cv2.putText(image, str(round(EAR, 2)), (10, 40),
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 1)


                    cv2.putText(image, f"FPS = {FPS}", (10, 100),
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 1)

                    cv2.putText(image, f"BLINK COUNT = {TOTAL}", (10, 130),
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 1)

                    # MINUTE NUMBER
                    minute_num = get_Minute(FRAME_COUNT)
                

                    FRAME_COUNT += 1
                    print(
                        f"FRAME COUNT = {FRAME_COUNT}, MINUTE = {minute_num}, EAR = {EAR}, IMAGE_SHAPE = {image.shape}, Video_Name={x}")

                    DATA.append([
                    FRAME_COUNT,
                    EAR_L,
                    EAR_R,
                    EAR,
                    EAR_diff_LAR,
                    newEAR,
                    # MAR,
                    # MOEAR
                    ])

                else:
                    print(results)
                    FRAME_COUNT+=1
                    EAR_L=0
                    EAR_R =0
                    EAR = 0
                    EAR_diff_LAR=0
                    newEAR=0
                    EAR_LIST.append(EAR)
                    DATA.append([
                        FRAME_COUNT,
                        EAR_L,
                        EAR_R,
                        EAR,
                        EAR_diff_LAR,
                        newEAR,
                        #MAR,
                        # MOEAR
                    ])
                #print(DEBUG)
                if DEBUG:
                    df = pd.DataFrame(DATA)
                    df.to_csv(f"./Feature_Extraction/{PERSON}/{NAME}_EAR.csv",mode="a", index=None, header=None)
                #image resize as it wsa not fit in single screen
                #image_Resize = cv2.resize(image, (1280,720))
                cv2.imshow('MediaPipe Face Mesh', image)
                if cv2.waitKey(5) & 0xFF == ord("q") or FRAME_COUNT >= length:
                    break

        cap.release()


if __name__ == "__main__":
    main()
