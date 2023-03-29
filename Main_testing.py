import cv2
import mediapipe as mp

mp_drawing = mp.solutions.face_mesh
mp_face_mesh = mp_drawing.FaceMesh()

# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    # if not success:
    #         print("Ignoring empty camera frame")
    #         continue
    height, width , _ = frame.shape
    # frame= cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    # frame.flags.writeable = False
    results = mp_face_mesh.process(frame)
    try:
        for face_landmarks in results.multi_face_landmarks:
            for i in range(0,468):
                landmark = face_landmarks.landmark[i]
                locx = int(landmark.x * width)
                locy = int(landmark.y * height)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                cv2.circle(frame, (locx,locy),1,(0,200,0),0)
                cv2.imshow("Image",frame)

    except:
        cv2.imshow('Face Mesh', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()