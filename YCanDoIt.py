import makedata
import train
import recognition


video_path = "얼굴 각도 다양.mp4"
img_path = "김인호.png"

cnt = makedata.make_data(video_path, img_path)
train.training(cnt)
recognition.reco(video_path)