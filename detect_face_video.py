from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="model file path")
ap.add_argument("-m", "--model", required=True, help="model file path")
ap.add_argument("-c", "--confidence", type=float, default=0.5)
args = vars(ap.parse_args())

print("[INFO] Model Loading")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] Video streaming")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# loop over the frames from video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    frame = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    # loop over detections
    for i in range(detections.shape[2]):
        # extract confidence score of predictions
        confidence = detections[0, 0, i, 2]
        if confidence < args['confidence']:
            continue
    
        # compute (x, y) coordinates of the bounding box
        box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
        (startX, startY, endX, endY) = box.astype('int')

        # draw bounding box
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        # draw rectangle
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        # Put Text on image
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 'q' breaks the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()