import cv2
import os


def capture_vehicle(frame, x1, y1, x2, y2, filename):
    if not os.path.exists("../vehicles"):
        os.makedirs("../vehicles")

    roi = frame[y1: y2, x1:x2]
    path = "../vehicles/" + filename
    print("captured " + filename)
    cv2.imwrite(path, roi)
