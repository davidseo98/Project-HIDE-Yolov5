import makedata
import train
import recognition_tracker


video_path = "유퀴즈.mp4"
img_path = "유재석.png"

print("making data...")
cnt = makedata.make_data(video_path, img_path)
print("training recognizer...")
train.training(cnt)
print("converting video...")
recognition_tracker.reco(video_path)