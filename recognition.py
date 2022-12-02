import cv2
import detect_simple
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()


def reco(video_path):
    recognizer.read('trainer.yml')
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    day = datetime.today().strftime("%Y%m%d")
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    output_movie = cv2.VideoWriter(f"video_"+str(day)+video_path, fourcc, fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = video.read()

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

                    if (confidence < 50):
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    else:
                        blur = frame[bottom:top, left:right]
                        blur = cv2.blur(blur,(50,50))
                        frame[bottom:top, left:right] = blur

        else: break

        output_movie.write(frame)
