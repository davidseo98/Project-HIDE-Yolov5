import makedata
import train
import recognition


video_path = "유퀴즈.mp4"
img_path = "유재석.png"

cnt = makedata.make_data(video_path, img_path)
train.training(cnt)
recognition.reco(video_path)