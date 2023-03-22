import pika
import sys
import os
import cv2
import uuid
import torch
from custom_utils.capture import capture_vehicle
import math
import dlib

INITIAL_PYTHON_QUEUE = 'INITIAL_PYTHON_QUEUE'
SECOND_PYTHON_QUEUE = "SECOND_PYTHON_QUEUE"
MAX_SPEED = 30

model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")

WIDTH = 1280
HEIGHT = 720


def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(
        location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 9.6
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed


def main():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue=INITIAL_PYTHON_QUEUE)

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)
        video_path = "../uploads/" + str(body.decode())
        print(video_path)
        if not os.path.exists(video_path):
            return

        rectangleColor = (0, 255, 0)
        frameCounter = 0
        currentCarID = 0

        carTracker = {}
        carLocation1 = {}
        carLocation2 = {}
        speed = [None] * 1000

        video = cv2.VideoCapture(video_path)
        ret, image = video.read()

        while ret:
            if type(image) == type(None):
                break
            image = cv2.resize(image, (WIDTH, HEIGHT))
            resultImage = image.copy()
            frameCounter = frameCounter + 1
            carIDtoDelete = []

            for carID in carTracker.keys():
                trackingQuality = carTracker[carID].update(image)
                if trackingQuality < 7:
                    carIDtoDelete.append(carID)

            for carID in carIDtoDelete:
                carTracker.pop(carID, None)
                carLocation1.pop(carID, None)
                carLocation2.pop(carID, None)

            if not (frameCounter % 10):
                result = model(image)
                df = result.pandas().xyxy[0]
                df = df.drop(['confidence', 'name'], axis=1)

                for (_x, _y, _xm, _ym, class_id) in df.values.astype(int):
                    x = (_x)
                    y = (_y)
                    xm = _xm
                    ym = _ym
                    w = xm-x
                    h = ym-y

                    x_cen = x + 0.5 * w
                    y_cen = y + 0.5 * h

                    matchCarID = None
                    if (x >= 10 and y >= 10):
                        for carID in carTracker.keys():
                            trackedPosition = carTracker[carID].get_position()

                            t_x = int(trackedPosition.left())
                            t_y = int(trackedPosition.top())
                            t_w = int(trackedPosition.width())
                            t_h = int(trackedPosition.height())

                            t_x_cen = t_x + 0.5 * t_w
                            t_y_cen = t_y + 0.5 * t_h

                            if ((t_x <= x_cen <= (t_x + t_w)) and (t_y <= y_cen <= (t_y + t_h)) and (x <= t_x_cen <= (x + w)) and (y <= t_y_cen <= (y + h))):
                                matchCarID = carID

                        if matchCarID is None:
                            tracker = dlib.correlation_tracker()
                            tracker.start_track(
                                image, dlib.rectangle(x, y, x + w, y + h))

                            carTracker[currentCarID] = tracker
                            carLocation1[currentCarID] = [x, y, w, h]

                            currentCarID = currentCarID + 1

            for carID in carTracker.keys():
                trackedPosition = carTracker[carID].get_position()

                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())

                cv2.rectangle(resultImage, (t_x, t_y),
                              (t_x + t_w, t_y + t_h), rectangleColor, 2)
                cv2.putText(resultImage, str(carID), (int(t_x - 50 + t_w/2),
                            int(t_y+t_h-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                carLocation2[carID] = [t_x, t_y, t_w, t_h]

            for i in carLocation1.keys():
                if frameCounter % 1 == 0:
                    [x1, y1, w1, h1] = carLocation1[i]
                    [x2, y2, w2, h2] = carLocation2[i]

                    carLocation1[i] = [x2, y2, w2, h2]

                    if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                        cv2.line(resultImage, (0, 275),
                                 (100, 275), (255, 255, 100), 2)
                        cv2.line(resultImage, (0, 285),
                                 (100, 285), (255, 255, 100), 2)
                        if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                            speed[i] = estimateSpeed(
                                [x1, y1, w1, h1], [x1, y2, w2, h2])

                        if speed[i] != None and y1 >= 70:
                            cv2.putText(resultImage, str(int(speed[i])) + "km/h", (int(
                                x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                            if int(speed[i]) > MAX_SPEED:
                                filename = str(uuid.uuid4()) + ".jpg"
                                capture_vehicle(
                                    resultImage, x2, y2, x2 + w2, y2 + h2, filename)
                                channel.basic_publish(
                                    exchange='', routing_key=SECOND_PYTHON_QUEUE, body=filename)
                                print(" [x] Sent " + filename)

            cv2.imshow('result', resultImage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, image = video.read()

        video.release()

    channel.basic_consume(queue=INITIAL_PYTHON_QUEUE,
                          on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
