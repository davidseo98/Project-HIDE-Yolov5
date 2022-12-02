import makedata
import train
import recognition_tracker


video_path = "test.mp4"
img_path = "test_img.png"

print("making data...")
cnt = makedata.make_data(video_path, img_path)
print("training recognizer...")
train.training(cnt)
print("converting video...")
recognition_tracker.reco(video_path)