import os
import numpy as np
import cv2
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "images"

def getImages(path):

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    IDs = []

    for image_path in image_paths:

        faces_img = Image.open(image_path).convert('L')
        face_np = np.array(faces_img, 'uint8')

        ID = int(image_path[11])
        print (ID)

        faces.append(face_np)
        IDs.append(ID)

        cv2.imshow("Adding Faces for Training", face_np)
        cv2.waitKey(1)

    return np.array(IDs), faces

IDs, faces = getImages(path)

recognizer.train(faces, IDs)
recognizer.write("train/trainingdata.yml")

cv2.destroyAllWindows
