import cv2
import detect_simple
from datetime import datetime
import dlib

recognizer = cv2.face.LBPHFaceRecognizer_create()

def check_person(pos, left, right, top, bottom):
    t_x, t_y, t_w, t_h = pos
    print("corrdinates", left, right, bottom, top)
    c_x, c_y, c_w, c_h = left, bottom, right - left, top - bottom
    print("tracker bounding box", t_x, t_y, t_w, t_h)
    print("current bounding box", c_x, c_y, c_w, c_h)
    start_x, start_y = max(t_x, c_x), max(t_y, c_y)
    end_x, end_y = min(t_x + t_w, c_x + c_w), min(t_y + t_h, c_y + c_h)
    if start_x > end_x or start_y > end_y : return 0
    print("overlap bounding box", start_x, start_y, end_x, end_y)
    box_size = (c_w * c_h)
    overlap_size = (end_x - start_x) * (end_y - start_y)
    return overlap_size / box_size 


def reco(video_path):
    recognizer.read('trainer.yml')
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    day = datetime.today().strftime("%Y%m%d")
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    output_movie = cv2.VideoWriter(f"video_"+str(day)+video_path, fourcc, fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    cnt = 1
    trackers = [10000]
    while True:
        ret, frame = video.read()

        if ret:
            print(f"converting frame {cnt}")
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
                        if (confidence > trackers[0]): continue
                        #tracker = dlib.correlation_tracker()
                        tracker = cv2.legacy_TrackerCSRT.create()
                        print(confidence, left, right, top, bottom)
                        #rect = dlib.rectangle(left, bottom, right, top)
                        x, y, w, h = left, bottom, right - left, top - bottom
                        past_x, past_y = x, y
                        tracker.init(frame, (x, y, w, h))
                        trackers = [confidence, tracker]

                    else:
                        if trackers[0] != 10000:
                            _, pos = trackers[1].update(frame)
                            overlap = check_person(pos, left, right, top, bottom)
                            print(overlap)
                            if overlap > 0.5: continue

                        blur = frame[bottom:top, left:right]
                        blur = cv2.blur(blur,(50,50))
                        frame[bottom:top, left:right] = blur

            if trackers[0] != 10000:
                cur_tracker = trackers[1]
                print(cur_tracker.update(frame))
                is_tracking, pos = cur_tracker.update(frame)
                
                cur_x, cur_y = pos[0], pos[1]

                if not is_tracking or (abs(cur_x - past_x) > 10 or abs(cur_y - past_y) > 10):
                    trackers=[10000]
                    continue

                pos = tuple(int(loc) for loc in pos)
                cv2.rectangle(frame, pos, (0, 0, 255), 2)
                past_x, past_y = cur_x, cur_y


        else: break
        cnt += 1
        output_movie.write(frame)