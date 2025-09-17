import os
import argparse
import imutils
import cv2
from codeConfig import social_distancing_config as config
from codeConfig.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
argp = argparse.ArgumentParser()
argp.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
argp.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
argp.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
arguments = vars(argp.parse_args())
lPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
Labels = open(lPath).read().strip().split("\n")
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
print("YOLO is being loaded....")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("Accessing video stream...")

#To view stock footage uncomment one of the line and comment line after camera mode

#vid = cv2.VideoCapture("pedestrians.mp4")
#vid = cv2.VideoCapture("nypedestrians.mp4")
#vid = cv2.VideoCapture("ffkk.mp4")
#vid = cv2.VideoCapture("production ID_4562551.mp4")

#Camera Mode
vid = cv2.VideoCapture(0)
wrt = None

while True:
	(grab, frame) = vid.read()

	if not grab:
		break

	frame = imutils.resize(frame, width=700)
	res = detect_people(frame, net, ln,
		personIdx=Labels.index("person"))

	violation = set()

	
	if len(res) >= 1:
		
		cen = np.array([r[2] for r in res])
		D = dist.cdist(cen, cen, metric="euclidean")
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				if D[i, j] < config.MIN_DISTANCE:
					violation.add(i)
					violation.add(j)
	for (i, (prob, bbox, centroid)) in enumerate(res):
		(X, Y, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)
		if i in violation:
			color = (0, 0, 255)
		cv2.rectangle(frame, (X, Y), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)
	text = "Violations: {}".format(len(violation))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
	if arguments["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		
	if arguments["output"] != "" and wrt is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		wrt = cv2.VideoWriter(arguments["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)
	if wrt is not None:
		wrt.write(frame)
cv2.waitKey(1)

