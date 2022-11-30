import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from time import time
from solutions import image_resize
from solutions import checkRotation, correct_rotation, save_Crop, clean_Folder
from solutions import get_Aspect_Ratios, get_Crops_L_Eye, get_Minute

# If false it wont save any data-points into a csv
DEBUG = True
SAVE = False
PREDICT = False
PERSON = "p1"

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3


def main() -> None:
    INPUT = input("""
    Enter the choice:
    1. Webcam
    2. Video Path
    """)

    #if DEBUG:
    #    NAME = input("Enter the name: ")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    FRAME_COUNT = 0
    rotateCode = None
    length = 1200
    prev_frame_time, new_frame_time = 0, 0
    COUNTER, TOTAL = 0, 0

    EAR_LIST = []

    # For webcam input:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    if INPUT == "1":
        cap = cv2.VideoCapture(0)
    else:
        video = input("Enter the video name:")
        path = f"./data/{PERSON}/{video}"
        print(path)
        cap = cv2.VideoCapture(path)

        # rotateCode = checkRotation(path)
        rotateCode = False

        length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
        print(f"LENGTH == {length}")

    # cap.set(3, WIDTH)
    # cap.set(4, HEIGHT)

    # Clean the folder and create a new file
    if DEBUG:
        clean_Folder(f"./data/DB_video_22.csv", f"DB_video_22.csv")

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

            # fix the image orientation
            if rotateCode:
                image = correct_rotation(image, rotateCode)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

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
                EAR_L, EAR_R,EAR,p1,p2,p3,p4,p5,p6 = get_Aspect_Ratios(person1_results, "EAR")
                MAR = get_Aspect_Ratios(person1_results, "MAR")
                MOEAR = MAR / EAR

                # Detection of blinks
                if EAR < EYE_AR_THRESH:
                    COUNTER += 1
                
                else:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1

                    COUNTER = 0

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

                cv2.putText(image, str(round(MOEAR, 2)), (10, 70),
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 1)

                cv2.putText(image, f"FPS = {FPS}", (10, 100),
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 1)

                cv2.putText(image, f"BLINK COUNT = {TOTAL}", (10, 130),
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 1)

                # MINUTE NUMBER
                minute_num = get_Minute(FRAME_COUNT)
                

                FRAME_COUNT += 1
                print(
                    f"FRAME COUNT = {FRAME_COUNT}, MINUTE = {minute_num}, EAR = {EAR}, IMAGE_SHAPE = {image.shape}")

                DATA.append([
                    FRAME_COUNT,
                    EAR_L,
                    EAR_R,
                    EAR,
                    p1,p2,p3,p4,p5,p6
                    #MAR,
                    #MOEAR,
                    #minute_num,
                    #int(round(time() * 1000)),
                ])

            else:
                FRAME_COUNT+=1
                EAR_L=0
                EAR_R =0
                EAR = 0
                DATA.append([
                    FRAME_COUNT,
                    EAR_L,
                    EAR_R,
                    EAR,
                    p1,p2,p3,p4,p5,p6
                    #MAR,
                    #MOEAR,
                    #minute_num,
                    #int(round(time() * 1000)),
                ])

            if DEBUG:
                df = pd.DataFrame(DATA)
                df.to_csv(f"./data/DB_video_22.csv",
                          mode="a", index=None, header=None)

            cv2.imshow('MediaPipe Face Mesh', image)
            if cv2.waitKey(5) & 0xFF == ord("q") or FRAME_COUNT >= length:
                break

    cap.release()


if __name__ == "__main__":
    main()
