import numpy as np
import cv2

# Built-in Camera
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("train/trainingdata.yml")

ID = -1
frame_counter = 0
face_counter = 0
# # Map user IDs to Names
dict = {}
# dict = {}
list = []

def overlayText(txt,x,y):
    cv2.putText(img, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX,1,255)

while True:
    cur_faces = 0
    frame_counter += 1
    if (frame_counter % 5) == 0:
        # cap.read() returns a boolean value, ret will be True if frame is read correctly
        ret, img = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x, y),(x+w, y+h),(255,0,0),2)
            ID,conf = rec.predict(gray[y:y+h, x:x+w])
            cur_faces = len(faces)

            if ID not in list:
                list.append(ID)
                face_counter += 1

            if conf < 45:
                if ID in dict.keys():
                    overlayText(str(dict[ID]), x, y)
                else:
                    overlayText(str(ID), x, y)

            else:
                overlayText(str(conf), x, y)

        # Displays number of unique faces seen
        overlayText("Total People Seen: " + str(face_counter), 920, 30)
        # Displays the number of faces currently in the screen
        overlayText("Current Faces: " + str(cur_faces), 920, 80)

        # Display the current frame
        cv2.imshow('Live Recording', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture
cap.release()
cv2.destroyAllWindows()
