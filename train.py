import cv2
import numpy as np
import face_recognition
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()

def getImagesAndLabels(cnt):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    
    for imagePath in imagePaths[:cnt]:
        image = face_recognition.load_image_file(imagePath)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faceSamples.append(image)
        ids.append(0)
        
    return faceSamples,ids

def training(cnt):
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(cnt)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))