import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

sample_n = 0

ID = input('Enter user id: ')

while True:
    # cap.read() returns a boolean value, will be True if frame is read correctly
    ret, img = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        sample_n += 1
        cv2.imwrite("images/User" + str(ID) + "." + str(sample_n) + ".jpg", gray[y:y+h, x:x+w])
        img = cv2.rectangle(img,(x, y),(x+w, y+h),(255,0,0),2)
        cv2.waitKey(500)
        # Display the current frame
    cv2.imshow('live', img)

    if cv2.waitKey(1) & 0xFF == ord('q') or sample_n > 39:
        break


# Release the capture
cap.release()
cv2.destroyAllWindows()
