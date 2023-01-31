import cv2
import uuid


def capture_vehicle(frame, x1, y1, x2, y2):
    roi = frame[y1: y2, x1:x2]
    filename = str(uuid.uuid4()) + ".jpg"
    path = "../vehicles/" + filename
    print("captured " + filename)
    cv2.imwrite(path, roi)

