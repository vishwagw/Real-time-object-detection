import cv2
import argparse
import time
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils.video import FPS

# argumen paersing:
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="./models/MobileNetSSD_deploy.caffemodel")
ap.add_argument("-m", "--model", required=True,
	help="./models/MobileNetSSD_deploy.prototxt.txt")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

