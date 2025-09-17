from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import time
import cv2
import math
from modules.detection import detect_people
from scipy.spatial import distance as dist
from modules.config import camera_no


Lpath = "yolo-coco\coco.names"
Labels = open(Lpath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 
                        255, 
                        size=(len(Labels), 3),
                        dtype="uint8")

weightsPath = "yolo-coco\yolov3.weights"
configPath = "yolo-coco\yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


confidence_threshold = 0.4


if True:
	
	print("")
	print("Searching for GPU")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)



print("loading face detector model...")
prototxtPath = "models/deploy.prototxt"
weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


model_store_dir= "models/classifier.model"
maskNet = load_model(model_store_dir)

cap = cv2.VideoCapture(0)       
while (cap.isOpened()):
    ret, image = cap.read()

    if ret == False:
        break

    image = cv2.resize(image, (720, 640))
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)

    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 
                                1/255.0, 
                                (416, 416), 
                                swapRB=True, 
                                crop=False)

    results = detect_people(image, net, ln,
                            personIdx=Labels.index("person"))
        

    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Time taken to predict the image: {:.6f}seconds".format(end-start))
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    ind = []
    for i in range(0, len(classIDs)):
        if (classIDs[i] == 0):
            ind.append(i)

    a = []
    b = []
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            a.append(x)
            b.append(y)
            #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    distance = []
    nsd = []
    for i in range(0, len(a) - 1):
        for k in range(1, len(a)):
            if (k == i):
                break
            else:
                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])
                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                distance.append(d)
                if (d <= 100.0):
                    nsd.append(i)
                    nsd.append(k)
                nsd = list(dict.fromkeys(nsd))

    color = (0, 0, 255)
    for i in nsd:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        text = "Alert"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    color = (138, 68, 38)
    

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (416, 416), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence =detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

          
            serious = set()
            abnormal = set()

           
            if len(results) >= 2:
               
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                       
                        if D[i, j] < 50:
                            
                            serious.add(i)
                            serious.add(j)
                        
                        if (D[i, j] < 80) and not serious:
                            abnormal.add(i)
                            abnormal.add(j)

            
            for (i, (prob, bbox, centroid)) in enumerate(results):
                
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

               
                if i in serious:
                    color = (0, 0, 255)
                elif i in abnormal:
                    color = (0, 255, 255) 

               
                cv2.circle(image, (cX, cY), 2, color, 2)


           
            (mask, without_mask) = maskNet.predict(face)[0]
            if mask > without_mask:
                     label = "Mask"
                     color = (0, 255, 0)
            else:
                     label = "No Mask"
                     color = (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)
            c = cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            x = cv2.rectangle(image, (startX, startY), (endX, endY), color, 1)
            text = "Total serious violations: {}".format(len(serious))
            

            text1 = "Total abnormal violations: {}".format(len(abnormal))
           

            print("End of classifier")

    
    cv2.imshow("Image",image)
 
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


