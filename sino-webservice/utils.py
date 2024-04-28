
def calculate_map(predictions, labels):
    def calculate_iou(box1, box2):
        # Convert box coordinates to (x1, y1, x2, y2) format
        box1 = [
            box1[0] - box1[2] / 2,
            box1[1] - box1[3] / 2,
            box1[0] + box1[2] / 2,
            box1[1] + box1[3] / 2
        ]
        box2 = [
            box2[0] - box2[2] / 2,
            box2[1] - box2[3] / 2,
            box2[0] + box2[2] / 2,
            box2[1] + box2[3] / 2
        ]

        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate intersection area
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate area of each box
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Calculate union area
        union_area = box1_area + box2_area - intersection_area

        # Calculate IOU
        iou = intersection_area / union_area
        return iou

    def calculate_ap(precision, recall):
        # Compute area under the precision-recall curve
        ap = 0
        for i in range(1, len(recall)):
            ap += (recall[i] - recall[i-1]) * precision[i]
        return ap

    average_precision = 0
    classes = set(label[0] for label in labels)

    for cls in classes:
        cls_predictions = [pred for pred in predictions if pred[5] == cls]
        cls_labels = [label for label in labels if label[0] == cls]

        true_positives = 0
        false_positives = 0
        false_negatives = len(cls_labels)

        precision = []
        recall = []
        sorted_predictions = sorted(cls_predictions, key=lambda x: x[4], reverse=True)

        for pred in sorted_predictions:
            max_iou = 0
            for label in cls_labels:
                iou = calculate_iou(pred[1:], label[1:])
                if iou > max_iou:
                    max_iou = iou
            if max_iou >= 0.5:
                true_positives += 1
                false_negatives -= 1
            else:
                false_positives += 1
            precision.append(true_positives / (true_positives + false_positives))
            recall.append(true_positives / (true_positives + false_negatives))

        ap = calculate_ap(precision, recall)
        average_precision += ap

    map_score = average_precision / len(classes)
    return map_score


def load_label_files(label_file_path):
    """
    Load label file containing ground truth bounding boxes.
    Each line of the file should have the format: class x_center y_center width height
    """
    labels = []
    with open(label_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            labels.append([class_id, x_center, y_center, width, height])
    return labels