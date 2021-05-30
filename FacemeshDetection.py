import cv2
import mediapipe as mp
import time


cTime = 0
pTime = 0
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)
draw_spec = mp_drawing.DrawingSpec()

while True:
    ret_val, image = cap.read()
    # convert image to RGB
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imageRGB)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACE_CONNECTIONS, draw_spec, draw_spec)
            # to loop all the 468 landmarks
            for id, landmark in enumerate(face_landmarks.landmark):
                # print(landmark)
                # convert landmark values in pixels
                ih, iw, _ = image.shape
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                print(id, x, y)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(image, f"FPS {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, cv2.COLOR_GRAY2BGR555)
    cv2.imshow("frame", image)
    cv2.waitKey(1)
