import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image path")
ap.add_argument("-p", "--prototxt", required=True, help="prototxt file path")
ap.add_argument("-m", "--model", required=True, help="model file path")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="prob value")
args = vars(ap.parse_args())

# load model
print("[INFO] Loading Model")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load image
print("[INFO] Loading Image")
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
image = cv2.resize(image, (300, 300))
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

print("[INFO] Detecting Object")

net.setInput(blob)
detections = net.forward()

# loop over detection
for i in range(detections.shape[2]):
    # prob associated with prediction
    confidence = detections[0, 0, i, 2]
    # select detection more than threshold
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the associated prob
        print("box: ", [startX, startY, endX, endY])
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show output image
cv2.imwrite("detection.jpg", image)
cv2.imshow("Output", image)
cv2.waitKey(0)