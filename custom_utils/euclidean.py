import math
from custom_utils.speed import calculate_speed

useful_classes = [2, 3, 5, 7]
class_names = {"2": "car", "3": "motorcycle", "5": "bus", "6": "truck"}


class EuclideanDistanceTracker:
    """
        used only for demo_3.py
        only works for results from  yolov5 model using pytorch
    """

    def __init__(self):
        self.detections = {}  # {"23": {box: [[], [], [], []], "class": 0}, ...
        self.max_distance = 20  # radius
        self.count = 0

    def update(self, results):
        final_response = {}

        new_boxes = []
        new_classes = []
        readable_results = results.pandas().xyxy  # xmin, ymin, xmax, ymax, confidence, class, name
        [result] = readable_results
        for i in range(0, len(result)):
            class_id = result['class'][i]
            if class_id not in useful_classes:
                continue
            xmin = int(result['xmin'][i])
            ymin = int(result['ymin'][i])
            xmax = int(result['xmax'][i])
            ymax = int(result['ymax'][i])
            new_boxes.append([xmin, ymin, xmax, ymax])
            new_classes.append(class_id)

        for i in range(0, len(new_boxes)):
            new_box = new_boxes[i]
            new_class_id = new_classes[i]
            new_center = ((new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2)
            was_detected_in_previous_frame = False
            for detection_id, info in self.detections.items():
                prev_center = ((info["box"][0] + info["box"][2]) / 2, (info["box"][1] + info["box"][3]) / 2)
                euclidean_distance = math.sqrt(math.pow(prev_center[0] - new_center[0], 2) + math.pow(prev_center[1] - new_center[1], 2))
                if euclidean_distance <= self.max_distance:
                    was_detected_in_previous_frame = True
                    final_response[detection_id] = {"box": new_box, "class": new_class_id, "speed": calculate_speed(euclidean_distance, 1)}
                    break

            if was_detected_in_previous_frame is False:
                self.count = self.count + 1
                new_id = str(self.count)
                final_response[new_id] = {"box": new_box, "class": new_class_id, "speed": None}

        self.detections = final_response.copy()
        return final_response
