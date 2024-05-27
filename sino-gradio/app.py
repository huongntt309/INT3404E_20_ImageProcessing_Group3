import os
import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    best_model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    # Load pre-trained fine-tuned YOLOv8 model and weights
    model = YOLO(best_model_path)

    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output


# Function to run object detection on an image using YOLOv8
def detect_and_visualize_objects(image):
    # Convert Gradio image object to PIL image
    image_pil = Image.fromarray(image.astype('uint8'), 'RGB')

    # Detect objects on the image
    boxes = detect_objects_on_image(image_pil)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image_pil)
    for box in boxes:
        x1, y1, x2, y2, label, _ = box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1), label, fill="green")

    # Convert PIL image back to numpy array
    image_with_boxes = np.array(image_pil)

    return image_with_boxes


# Gradio interface
gr.Interface(fn=detect_and_visualize_objects, inputs="image", outputs="image").launch(share=True)