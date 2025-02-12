# for detecting in videos:
import cv2
from ultralytics import YOLO

# model load:
y_model = YOLO('./model/yolov8n.pt')

# video path:
#video_path = './input/input4.mp4'
# load video
cap = cv2.VideoCapture(1)
