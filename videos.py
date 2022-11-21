import cv2
import detect_simple
import face_recognition
import numpy as np
import cv2 as cv
import dlib
# Open the input movie file
input_movie = cv2.VideoCapture("곽튜브2.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
fps = input_movie.get(cv2.CAP_PROP_FPS)
print(length)

fourcc = cv2.VideoWriter_fourcc(*'FMP4')
output_movie = cv2.VideoWriter(f'video_20221121_G.mp4', fourcc, fps, (int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH)),int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))

lmm_image = face_recognition.load_image_file("곽튜브.png")
lmm_image = cv.cvtColor(lmm_image,cv.COLOR_BGR2RGB)
fc_loc = detect_simple.detect(lmm_image)
(left, bottom, right, top) = fc_loc[0].numpy()
left = int(left)
right = int(right)
top = int(top)
bottom = int(bottom)
lmm_face_encoding = face_recognition.face_encodings(lmm_image, [[top, right, bottom, left]])[0]
#lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]
font = cv2.FONT_HERSHEY_DUPLEX
# Initialize some variables

frame_number = 0

known_faces = [
    lmm_face_encoding
]

while frame_number < 1035:
    
    # Grab a single frame of video
    ret, frame = input_movie.read()
    if frame_number < 950:
        frame_number += 1
        continue
    frame_number += 1
    # Quit when the input video file ends
    face_locations = []
    face_encodings = []
    face_names = []
    
    
    if ret:
        #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Find all the faces and face encodings in the current frame of video
        det = detect_simple.detect(frame)
        
        
        # Label the results
        # [xmin, ymin, xmax, ymax, confidence, class]
        # yolov5
        if len(det) > 0:
            for tensor in det:
                (left, bottom, right, top) = tensor.numpy()
                left = int(left)
                right = int(right)
                top = int(top)
                bottom = int(bottom)
                
                face_locations.append([top, right, bottom, left])
                print(tensor.numpy())
                #
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            dist = face_recognition.face_distance(face_encodings, known_faces[0])
            index = np.argmin(dist)
            print(dist, index)
            if dist[index] < 0.4:
                (top, right, bottom, left) = face_locations[index]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, "find", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            else:
                (top, right, bottom, left) = face_locations[index]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
            for i in range(len(face_locations)):
                if i == index: continue
                cv2.rectangle(frame, (face_locations[i][3], face_locations[i][0]), (face_locations[i][1], face_locations[i][2]), (0, 0, 255), 3)

    else: break

    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()