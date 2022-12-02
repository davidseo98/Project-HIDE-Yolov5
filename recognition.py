import cv2
import detect_simple
from datetime import datetime

def reco(video_path, flag = 1):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    day = datetime.today().strftime("%Y%m%d")
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    output_movie = cv2.VideoWriter(f"video2_"+str(day)+video_path, fourcc, fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    frame_number = 0
    while True:
        ret, frame = video.read()

        print("{}/{}".format(frame_number, length))
        frame_number += 1

        match_faces = []
        dismatch_faces = []
        mosiac = []

        if ret:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            rgb_small_frame = frame[:, :, ::-1]
            det = detect_simple.detect(rgb_small_frame)

            if len(det) > 0:
                for tensor in det:
                    (left, bottom, right, top) = tensor.numpy()
                    left = int(left)
                    right = int(right)
                    top = int(top)
                    bottom = int(bottom)
                    _, confidence = recognizer.predict(gray[bottom:top, left:right])

                    # Check if confidence is less them 100 ==> "0" is perfect match

                    if confidence < 75:
                        match_faces.append([top, right, bottom, left])
                        """cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, str(confidence), (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,0), 1)"""
                    else:
                        dismatch_faces.append([top, right, bottom, left])
                        """blur = frame[bottom:top, left:right]
                        blur = cv2.blur(blur,(50,50))
                        frame[bottom:top, left:right] = blur
                        cv2.putText(frame, str(confidence), (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,0), 1)"""

        else: break

        if flag == 0:
            mosiac = match_faces
        else:
            mosiac = dismatch_faces

        for face in mosiac:
            (top, right, bottom, left) = face
            blur = frame[bottom:top, left:right]
            blur = cv2.blur(blur,(50,50))
            frame[bottom:top, left:right] = blur

        output_movie.write(frame)

