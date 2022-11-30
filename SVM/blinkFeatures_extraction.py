'''
This code extract blink features from video data

Features : [blink frequency, blink amplitude, blink duration , blink velocity]

Facial landmarks are extracted using MediaPipe's FaceMesh
'''

from __future__ import print_function
from distutils.log import debug
from cv2 import rotate

from scipy.spatial import distance as dist
import scipy.ndimage.filters as signal

from imutils import face_utils

import datetime
import imutils
# import dlib

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import*
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage.interpolation import shift
import pickle
from queue import Queue

import numpy as np
import cv2

import mediapipe as mp
import math
import itertools
import time
import shutup
import sys
import ffmpeg
import os

shutup.please()  ## to suppress pip warnings


# this "adjust_gamma" function directly taken from : https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def get_coordinates(face_landmarks, point):
    '''
    Extract 2D normalized coordinates using landmark index
    '''
    return (face_landmarks.landmark[point].x, face_landmarks.landmark[point].y)

def valid(value):
    '''
    Validate the extracted points
    '''
    return (value > 0 or math.isclose(0, value) and (value < 1 or math.isclose(1, value)))

def denormalize(points, shape):
    '''
    Denormalize coordinates based on frame size
    '''
    if not valid(points[0]) and valid(points[1]):
        return None
    
    x = min(math.floor(points[0] * shape[0]), shape[0] -1)
    y = min(math.floor(points[1] * shape[1]), shape[1] - 1)

    return (x, y)

def distance(p1, p2, size):
    '''
    Calculates euclidean distance
    '''
    p1_denorm = denormalize(p1, size)
    p2_denorm = denormalize(p2, size)
    return math.sqrt((p1_denorm[0] - p2_denorm[0]) ** 2 + (p1_denorm[1] - p2_denorm[1]) ** 2)

#def eye_aspect_ratio(eye):
#    # compute the euclidean distances between the two sets of
#    # vertical eye landmarks (x, y)-coordinates
#    A = dist.euclidean(eye[1], eye[5])
#    B = dist.euclidean(eye[2], eye[4])
#
#    # compute the euclidean distance between the horizontal
#    # eye landmark (x, y)-coordinates
#    C = dist.euclidean(eye[0], eye[3])
#
#    if C<0.1:           #practical finetuning due to possible numerical issue as a result of optical flow
#        ear=0.3
#    else:
#        # compute the eye aspect ratio
#        ear = (A + B) / (2.0 * C)
#    if ear>0.45:        #practical finetuning due to possible numerical issue as a result of optical flow
#        ear=0.45
#    # return the eye aspect ratio
#    return ear
#
#
#def mouth_aspect_ratio(mouth):
#
#    A = dist.euclidean(mouth[14], mouth[18])
#
#   C = dist.euclidean(mouth[12], mouth[16])
#
#   if C<0.1:           #practical finetuning
#        mar=0.2
#    else:
#        # compute the mouth aspect ratio
#        mar = (A ) / (C)
#
#   # return the mouth aspect ratio
#   return mar

def aspect_feature(pairs, face_landmarks, size):
    '''
    Calculates aspect ratio for set of face_landmarks
    '''
    vertical = 0
    horizontal = 0
    p1, p2 = pairs[1]
    p1_norm = get_coordinates(face_landmarks, p1)
    p2_norm = get_coordinates(face_landmarks, p2)
    vertical = distance(p1_norm, p2_norm, size)
        
    h1, h2 = pairs[0]
    h1_norm = get_coordinates(face_landmarks, h1)
    h2_norm = get_coordinates(face_landmarks, h2)
    horizontal = distance(h1_norm, h2_norm, size)

    if horizontal==0:
        return None

    return (vertical / horizontal)


def EMERGENCY(ear, COUNTER):
    '''
    Emergency counter for checking if person is asleep
    '''
    if ear < 0.21:
        COUNTER += 1

        if COUNTER >= 50:
            print('EMERGENCY SITUATION (EYES TOO LONG CLOSED)')
            print(COUNTER)
            COUNTER = 0
    else:
        COUNTER=0
    return COUNTER


def Linear_Interpolate(start,end,N):
    m=(end-start)/(N+1)
    x=np.linspace(1,N,N)
    y=m*(x-0)+start
    return list(y)


    
def correct_rotation(frame, rC):
    return cv2.rotate(frame, rC) 

def checkRotation(file_name):
    try:
        print(f"Probing video: {file_name} for rotations")
        metaData = ffmpeg.probe(file_name)
    except:
        return None

    stream_data = metaData.get('streams', [dict(tags = dict())])
    
    rC = None
    rotateCode = None
    
    for i in range(len(stream_data)):
        rotate=stream_data[i].get('tags',dict()).get('rotate', None)
        if rotate:
            rotateCode = int(rotate)

   
    if rotateCode in [90, -90, 270, -270]:
        rC = 1
        print(f"Found rotation: {rotateCode}")
    if rotateCode == 180:
        rC = None

    return rC

def blink_detector(output_textfile,input_video, debug):

    queue_frames = Queue(maxsize=7)

    FRAME_MARGIN_BTW_2BLINKS=3
    MIN_AMPLITUDE=0.04
    MOUTH_AR_THRESH=0.35
    MOUTH_AR_THRESH_ALERT=0.30
    MOUTH_AR_CONSEC_FRAMES=20

    EPSILON=0.01  # for discrete derivative (avoiding zero derivative)
    class Blink():
        def __init__(self):

            self.start=0 #frame
            self.startEAR=1
            self.peak=0  #frame
            self.peakEAR = 1
            self.end=0   #frame
            self.endEAR=0
            self.amplitude=(self.startEAR+self.endEAR-2*self.peakEAR)/2
            self.duration = self.end-self.start+1
            self.EAR_of_FOI=0 #FrameOfInterest
            self.values=[]
            self.velocity=0  #Eye-closing velocity




    def Ultimate_Blink_Check():
        #Given the input "values", retrieve blinks and their quantities
        retrieved_blinks=[]
        MISSED_BLINKS=False
        values=np.asarray(Last_Blink.values)
        THRESHOLD=0.4*np.min(values)+0.6*np.max(values)   # this is to split extrema in highs and lows
        N=len(values)
        Derivative=values[1:N]-values[0:N-1]    #[-1 1] is used for derivative
        i=np.where(Derivative==0)
        if len(i[0])!=0:
            for k in i[0]:
                if k==0:
                    Derivative[0]=-EPSILON
                else:
                    Derivative[k]=EPSILON*Derivative[k-1]
        M=N-1    #len(Derivative)
        ZeroCrossing=Derivative[1:M]*Derivative[0:M-1]
        x = np.where(ZeroCrossing < 0)
        xtrema_index=x[0]+1
        XtremaEAR=values[xtrema_index]
        Updown=np.ones(len(xtrema_index))        # 1 means high, -1 means low for each extremum
        Updown[XtremaEAR<THRESHOLD]=-1           #this says if the extremum occurs in the upper/lower half of signal
        #concatenate the beginning and end of the signal as positive high extrema
        Updown=np.concatenate(([1],Updown,[1]))
        XtremaEAR=np.concatenate(([values[0]],XtremaEAR,[values[N-1]]))
        xtrema_index = np.concatenate(([0], xtrema_index,[N - 1]))
        ##################################################################

        Updown_XeroCrossing = Updown[1:len(Updown)] * Updown[0:len(Updown) - 1]
        jump_index = np.where(Updown_XeroCrossing < 0)
        numberOfblinks = int(len(jump_index[0]) / 2)
        selected_EAR_First = XtremaEAR[jump_index[0]]
        selected_EAR_Sec = XtremaEAR[jump_index[0] + 1]
        selected_index_First = xtrema_index[jump_index[0]]
        selected_index_Sec = xtrema_index[jump_index[0] + 1]
        if numberOfblinks>1:
            MISSED_BLINKS=True
        if numberOfblinks ==0:
            if debug:
                print(Updown,Last_Blink.duration)
                print(values)
                print(Derivative)
        for j in range(numberOfblinks):
            detected_blink=Blink()
            detected_blink.start=selected_index_First[2*j]
            detected_blink.peak = selected_index_Sec[2*j]
            detected_blink.end = selected_index_Sec[2*j + 1]

            detected_blink.startEAR=selected_EAR_First[2*j]
            detected_blink.peakEAR = selected_EAR_Sec[2*j]
            detected_blink.endEAR = selected_EAR_Sec[2*j + 1]

            detected_blink.duration=detected_blink.end-detected_blink.start+1
            detected_blink.amplitude=0.5*(detected_blink.startEAR-detected_blink.peakEAR)+0.5*(detected_blink.endEAR-detected_blink.peakEAR)
            detected_blink.velocity=(detected_blink.endEAR-selected_EAR_First[2*j+1])/(detected_blink.end-selected_index_First[2*j+1]+1) #eye opening ave velocity
            retrieved_blinks.append(detected_blink)



        return MISSED_BLINKS,retrieved_blinks



    def Blink_Tracker(EAR,IF_Closed_Eyes,Counter4blinks,TOTAL_BLINKS,skip):
        BLINK_READY=False
        #If the eyes are closed
        if int(IF_Closed_Eyes)==1:
            Current_Blink.values.append(EAR)
            Current_Blink.EAR_of_FOI=EAR      #Save to use later
            if Counter4blinks>0:
                skip = False
            if Counter4blinks==0:
                Current_Blink.startEAR=EAR    #EAR_series[6] is the EAR for the frame of interest(the middle one)
                Current_Blink.start=reference_frame-6   #reference-6 points to the frame of interest which will be the 'start' of the blink
            Counter4blinks += 1
            if Current_Blink.peakEAR>=EAR:    #deciding the min point of the EAR signal
                Current_Blink.peakEAR =EAR
                Current_Blink.peak=reference_frame-6
        # otherwise, the eyes are open in this frame
        else:
            if Counter4blinks <2 and skip==False :           # Wait to approve or reject the last blink
                if Last_Blink.duration>15:
                    FRAME_MARGIN_BTW_2BLINKS=8
                else:
                    FRAME_MARGIN_BTW_2BLINKS=1
                if ( (reference_frame-6) - Last_Blink.end) > FRAME_MARGIN_BTW_2BLINKS:
                    # Check so the prev blink signal is not monotonic or too small (noise)
                    if  Last_Blink.peakEAR < Last_Blink.startEAR and Last_Blink.peakEAR < Last_Blink.endEAR and Last_Blink.amplitude>MIN_AMPLITUDE and Last_Blink.start<Last_Blink.peak:
                        if((Last_Blink.startEAR - Last_Blink.peakEAR)> (Last_Blink.endEAR - Last_Blink.peakEAR)*0.25 and (Last_Blink.startEAR - Last_Blink.peakEAR)*0.25< (Last_Blink.endEAR - Last_Blink.peakEAR)): # the amplitude is balanced
                            BLINK_READY = True
                            #####THE ULTIMATE BLINK Check

                            Last_Blink.values=signal.convolve1d(Last_Blink.values, [1/3.0, 1/3.0,1/3.0],mode='nearest')
                            # Last_Blink.values=signal.median_filter(Last_Blink.values, 3, mode='reflect')   # smoothing the signal
                            [MISSED_BLINKS,retrieved_blinks]=Ultimate_Blink_Check()
                            #####
                            TOTAL_BLINKS =TOTAL_BLINKS+len(retrieved_blinks)  # Finally, approving/counting the previous blink candidate
                            ###Now You can count on the info of the last separate and valid blink and analyze it
                            Counter4blinks = 0
                            print("MISSED BLINKS= {}".format(len(retrieved_blinks)))
                            return retrieved_blinks,int(TOTAL_BLINKS),Counter4blinks,BLINK_READY,skip
                        else:
                            skip=True
                            print('rejected due to imbalance')
                    else:
                        skip = True
                        print('rejected due to noise,magnitude is {}'.format(Last_Blink.amplitude))
                        print(Last_Blink.start<Last_Blink.peak)

            # if the eyes were closed for a sufficient number of frames (2 or more)
            # then this is a valid CANDIDATE for a blink
            if Counter4blinks >1:
                Current_Blink.end = reference_frame - 7  #reference-7 points to the last frame that eyes were closed
                Current_Blink.endEAR=Current_Blink.EAR_of_FOI
                Current_Blink.amplitude = (Current_Blink.startEAR + Current_Blink.endEAR - 2 * Current_Blink.peakEAR) / 2
                Current_Blink.duration = Current_Blink.end - Current_Blink.start + 1

                if Last_Blink.duration>15:
                    FRAME_MARGIN_BTW_2BLINKS=8
                else:
                    FRAME_MARGIN_BTW_2BLINKS=1
                if (Current_Blink.start-Last_Blink.end )<=FRAME_MARGIN_BTW_2BLINKS+1:  #Merging two close blinks
                    print('Merging...')
                    frames_in_between=Current_Blink.start - Last_Blink.end-1
                    print(Current_Blink.start ,Last_Blink.end, frames_in_between)
                    valuesBTW=Linear_Interpolate(Last_Blink.endEAR,Current_Blink.startEAR,frames_in_between)
                    Last_Blink.values=Last_Blink.values+valuesBTW+Current_Blink.values
                    Last_Blink.end = Current_Blink.end            # update the end
                    Last_Blink.endEAR = Current_Blink.endEAR
                    if Last_Blink.peakEAR>Current_Blink.peakEAR:  #update the peak
                        Last_Blink.peakEAR=Current_Blink.peakEAR
                        Last_Blink.peak = Current_Blink.peak
                        #update duration and amplitude
                    Last_Blink.amplitude = (Last_Blink.startEAR + Last_Blink.endEAR - 2 * Last_Blink.peakEAR) / 2
                    Last_Blink.duration = Last_Blink.end - Last_Blink.start + 1
                else:                                             #Should not Merge (a Separate blink)

                    Last_Blink.values=Current_Blink.values        #update the EAR list


                    Last_Blink.end = Current_Blink.end            # update the end
                    Last_Blink.endEAR = Current_Blink.endEAR

                    Last_Blink.start = Current_Blink.start        #update the start
                    Last_Blink.startEAR = Current_Blink.startEAR

                    Last_Blink.peakEAR = Current_Blink.peakEAR    #update the peak
                    Last_Blink.peak = Current_Blink.peak

                    Last_Blink.amplitude = Current_Blink.amplitude
                    Last_Blink.duration = Current_Blink.duration




            # reset the eye frame counter
            Counter4blinks = 0
        retrieved_blinks=0
        return retrieved_blinks,int(TOTAL_BLINKS),Counter4blinks,BLINK_READY,skip


    COUNTER = 0
    MCOUNTER=0
    TOTAL = 0
    MTOTAL=0
    TOTAL_BLINKS=0
    Counter4blinks=0
    skip=False # to make sure a blink is not counted twice in the Blink_Tracker function
    Last_Blink=Blink()

    # print("[INFO] loading facial landmark predictor...")
    # detector = dlib.get_frontal_face_detector()
    # # Load the Facial Landmark Detector
    # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # #Load the Blink Detector
    loaded_svm = pickle.load(open('./data/Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav', 'rb'))  # Trained SVM for detecting blinks
    # # grab the indexes of the facial landmarks for the left and
    # # right eye, respectively
    # (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    # print("[INFO] starting video stream thread...")

    mp_face_mesh = mp.solutions.face_mesh
    # face_mesh = mp_face_mesh.FaceMesh(max_num_faces = 1,refine_landmarks = True,min_detection_confidence = 0.5,min_tracking_confidence = 0.5) 

    left_eye_indexes = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
    right_eye_indexes = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
    # lip_indexes = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
    lip_indexes = [0, 267, 269, 270, 409, 91, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]


    right_eye_pairs = [[33, 133], [159, 145]]
    left_eye_pairs = [[362, 263], [386, 374]]
    lip_pairs = [[61, 91], [0, 17]]



    lk_params=dict( winSize  = (13,13),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    EAR_series=np.zeros([13])
    Frame_series=np.linspace(1,13,13)
    reference_frame=0
    First_frame=True

    if debug:
        top = tk.Tk()
        frame1 = Frame(top)
        frame1.grid(row=0, column=0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_frame =FigureCanvasTkAgg(fig, master=frame1)
        plot_frame.get_tk_widget().pack(side=tk.BOTTOM, expand=True)
        plt.ylim([0.0, 0.5])
        line, = ax.plot(Frame_series,EAR_series)
        plot_frame.draw()

    # loop over frames from the video stream


    stream = cv2.VideoCapture(input_video)

    
    total_count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

    rotateCode = checkRotation(input_video)

    frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    size = (frame_width, frame_height)

    print(size)
    start = datetime.datetime.now()
    number_of_frames=0
    with mp_face_mesh.FaceMesh(
            max_num_faces = 1,
            refine_landmarks = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5) as face_mesh:

        while stream.isOpened():
            grabbed, frame = stream.read()


            if not grabbed:
                print('not grabbed')
                print(number_of_frames)
                break

            if rotateCode:
                frame = correct_rotation(frame, rotateCode)


            # frame = imutils.resize(frame, width=450)

            # To Rotate by 90 degreees
            # rows=np.shape(frame)[0]
            # cols = np.shape(frame)[1]
            # M = cv2.getRotationMatrix2D((cols / 2, rows / 2),-90, 1)
            # frame = cv2.warpAffine(frame, M, (cols, rows))
            # queue_frames.put(frame)


            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #Brighten the image(Gamma correction)
            reference_frame = reference_frame + 1
            # # gray=adjust_gamma(gray,gamma=1.5)
            
            end = datetime.datetime.now()
            ElapsedTime=(end - start).total_seconds()

            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            rects = face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # cv2.namedWindow("KeyPoint Frame", cv2.WINDOW_NORMAL)
            frame.flags.writeable = True

            # cv2.imshow("KeyPoint Frame", frame)
                
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    


            if rects.multi_face_landmarks:
                queue_frames.put(frame)
                for face_landmarks in rects.multi_face_landmarks:
                    MAR = aspect_feature(lip_pairs, face_landmarks, size)
                    leftEAR = aspect_feature(left_eye_pairs, face_landmarks, size)
                    rightEAR = aspect_feature(right_eye_pairs, face_landmarks, size)
                    
                    for lei in left_eye_indexes:
                        points  = (face_landmarks.landmark[lei].x, face_landmarks.landmark[lei].y)
                        cv2.circle(frame, denormalize(points, size), 1, (0, 0, 255))

                    for rei in right_eye_indexes:
                        points  = (face_landmarks.landmark[rei].x, face_landmarks.landmark[rei].y)
                        cv2.circle(frame, denormalize(points, size), 1, (0, 0, 255))

                    for lip in lip_indexes:
                        points  = (face_landmarks.landmark[lip].x, face_landmarks.landmark[lip].y)
                        cv2.circle(frame, denormalize(points, size), 1, (0, 0, 255))
                    
                    Mouth = np.array([(face_landmarks.landmark[point].x, face_landmarks.landmark[point].y) for point in list(itertools.chain(*lip_pairs))])
                    leftEye = np.array([(face_landmarks.landmark[point].x, face_landmarks.landmark[point].y) for point in list(itertools.chain(*left_eye_pairs))])
                    rightEye = np.array([(face_landmarks.landmark[point].x, face_landmarks.landmark[point].y) for point in list(itertools.chain(*right_eye_pairs))])



                number_of_frames = number_of_frames + 1  # we only consider frames that face is detected
                First_frame = False

                ###############YAWNING##################

                if MAR > MOUTH_AR_THRESH:
                    MCOUNTER += 1

                elif MAR < MOUTH_AR_THRESH_ALERT:

                    if MCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                        MTOTAL += 1

                    MCOUNTER = 0

                ##############YAWNING####################


                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                # EAR_series[reference_frame]=ear
                EAR_series = shift(EAR_series, -1, cval=ear)

                ############HANDLING THE EMERGENCY SITATION################
                COUNTER=EMERGENCY(ear,COUNTER)

                # EMERGENCY SITUATION (EYES TOO LONG CLOSED) ALERT THE DRIVER IMMEDIATELY
                ############HANDLING THE EMERGENCY SITATION################
                if queue_frames.full() and (reference_frame>15):  #to make sure the frame of interest for the EAR vector is int the mid
                    EAR_table = EAR_series
                    IF_Closed_Eyes = loaded_svm.predict(EAR_series.reshape(1,-1))
                    if Counter4blinks==0:
                        Current_Blink = Blink()
                    retrieved_blinks, TOTAL_BLINKS, Counter4blinks, BLINK_READY, skip = Blink_Tracker(EAR_series[6],
                                                                                                        IF_Closed_Eyes,
                                                                                                        Counter4blinks,
                                                                                                        TOTAL_BLINKS, skip)
                    if (BLINK_READY==True):
                        reference_frame=20   #initialize to a random number to avoid overflow in large numbers
                        skip = True
                        #####
                        BLINK_FRAME_FREQ = TOTAL_BLINKS / number_of_frames
                        for detected_blink in retrieved_blinks:
                            print(detected_blink.amplitude, Last_Blink.amplitude)
                            print(detected_blink.duration, detected_blink.velocity)
                            print('-------------------')

                            if(detected_blink.velocity>0):
                                with open(output_file, 'ab') as f_handle:
                                    f_handle.write(b'\n')
                                    np.savetxt(f_handle,[TOTAL_BLINKS,BLINK_FRAME_FREQ*100,detected_blink.amplitude,detected_blink.duration,detected_blink.velocity, MAR, ear], delimiter=', ', newline=' ',fmt='%.4f')





                        Last_Blink.end = -10 # re initialization
                        #####

                    line.set_ydata(EAR_series)
                    plot_frame.draw()
                    frameMinus7=queue_frames.get()
                elif queue_frames.full():         #just to make way for the new input of the queue_frames when the queue_frames is full
                    junk =  queue_frames.get()

            if debug:
                cv2.namedWindow("KeyPoint Frame", cv2.WINDOW_NORMAL)

                cv2.imshow("KeyPoint Frame", frame)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    # do a bit of cleanup
    stream.release()
    # cv2.destroyAllWindows()



def cap_progress(current_frame, total_frames):
    msg = "Processing: %d%% [%d / %d] frames" % (current_frame / total_frames * 100, current_frame, total_frames)
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()



if __name__ == '__main__':

    root_dir = "/home/anshal/work/Mahindra/drowsiness/data/p1"
    debug = True

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path[-3:] in ["mov", "MOV", "mp4", "MP4", "m4v"]:   # check for correct extension
                tag = file_path.split(".")[0]

                output_file = tag + "_check.txt"
                if os.path.exists(os.path.join(root, output_file)):
                    print(f"Skipping file: {os.path.join(root, output_file)}")
                    continue
                print(f"Processing file: {file_path}")
                start = time.time()
                blink_detector(output_file, file_path, debug)
                print(f"\n Time taken to process: {(time.time() - start)/60} min")
