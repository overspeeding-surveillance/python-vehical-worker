import pika, sys, os
import cv2
import torch
from custom_utils.euclidean import EuclideanDistanceTracker
from custom_utils.capture import capture_vehicle

INITIAL_PYTHON_QUEUE = 'INITIAL_PYTHON_QUEUE'
MAX_SPEED = 50

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

tracker = EuclideanDistanceTracker()


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue=INITIAL_PYTHON_QUEUE)

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)
        video_path = "../uploads/" + str(body.decode())
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()

        while ret:
            results = model(frame)

            if not ret:
                ret, frame = cap.read()
                continue

            response = tracker.update(results)

            for vehicle_id, info in response.items():  # info: {'box': [[], ...], 'class': 1}
                x1 = info['box'][0]
                y1 = info['box'][1]
                x2 = info['box'][2]
                y2 = info['box'][3]
                speed = info['speed']

                if speed and speed > MAX_SPEED:
                    capture_vehicle(frame, x1, y1, x2, y2)

            ret, frame = cap.read()

        cap.release()

    channel.basic_consume(queue=INITIAL_PYTHON_QUEUE, on_message_callback=callback, auto_ack=True)

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

