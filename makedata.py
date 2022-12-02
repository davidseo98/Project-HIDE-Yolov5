import cv2
import detect_simple
import face_recognition

# Open the input movie file

def make_data(video_path, img_path):
    video = cv2.VideoCapture(video_path)

    lmm_image = face_recognition.load_image_file(img_path)
    lmm_image = cv2.cvtColor(lmm_image,cv2.COLOR_BGR2RGB)
    fc_loc = detect_simple.detect(lmm_image)

    (left, bottom, right, top) = fc_loc[0].numpy()
    left = int(left)
    right = int(right)
    top = int(top)
    bottom = int(bottom)
    img_trim = lmm_image[bottom - int(bottom*0.1):top + int(top*0.1), left - int(left*0.1):right + int(right*0.1)]
    lmm_face_encoding = face_recognition.face_encodings(img_trim, model = 'large')[0]

    known_face_encodings = [lmm_face_encoding]

    save_flag = True
    frame_number = 0
    cnt = 0
    while True:
        ret, frame = video.read()
        frame_number += 1
        if frame_number % 50 == 1:
            save_flag = True
        if ret:
            rgb_small_frame = frame[:, :, ::-1]
            
            det = detect_simple.detect(rgb_small_frame)
            # yolo로 얼굴 detect하고 해당 좌표로 frame을 자르고 해당 이미지 저장
            if len(det) > 0:
                for tensor in det:
                    (left, bottom, right, top) = tensor.numpy()
                    left = int(left)
                    right = int(right)
                    top = int(top)
                    bottom = int(bottom)
                    img_trim = frame[bottom - int(bottom*0.1):top + int(top*0.1), left - int(left*0.1):right + int(right*0.1)]
                    face_encoding = face_recognition.face_encodings(img_trim, model = 'large')
                    
                    if len(face_encoding) > 0 and save_flag == True:
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding[0])
                        if face_distances[0] < 0.6:
                            print("cnt: {}".format(cnt))
                            cv2.imwrite("dataset/data" + str(cnt) + ".png", frame[bottom:top, left:right])
                            save_flag = False
                            cnt += 1
                        
    
                        
        else: break

    return cnt