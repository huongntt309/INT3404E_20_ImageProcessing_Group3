import os
from ultralytics import YOLO
from flask import jsonify, request, Response, Flask, template_rendered
from waitress import serve
from PIL import Image
import json

from utils import load_label_files

app = Flask(__name__)
model = None

def set_up():
  global model
  file_dir = os.path.dirname(os.path.realpath(__file__))
  model_path = os.path.join(file_dir, "best.pt")
  model = YOLO(model_path)
  
  print("Running at: http://localhost:8080")

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    file_dir = os.path.dirname(os.path.realpath(__file__))
    html_path = os.path.join(file_dir, "index.html")
    with open(html_path) as file:
        return file.read()

# TODO: calculate mAP for this list 
          # boxes have predictions    x_center, y_center, w, h, class, confidence score
          # label_file have ground truth  class, x_center, y_center, w, h
          # and just have 1 class is 0, how can i calculate mAP
@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded files with names "image_files" and "label_files", 
    passes them through your detection function 
    and returns an array of bounding boxes for each image.
    :return: a JSON array of objects bounding 
    boxes in format 
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    # Get uploaded files
    image_files = request.files.getlist("image_files")
    label_files = request.files.getlist("label_files")

    # Ensure the number of image files and label files match
    if len(image_files) != len(label_files):
        return jsonify({"error": "Number of image files and label files do not match"}), 400

    # Perform detection for each pair of image and label
    all_boxes = []
    for img_file in image_files:
        # Extract filename without extension
        img_filename = os.path.splitext(img_file.filename)[0]
        print("labels: ", img_filename)
        
        # Find corresponding label file
        lbl_filename = img_filename + ".txt"
        
        lbl_file = None
        for file in label_files:
            if file.filename == lbl_filename:
                lbl_file = file
                break
        
        # Perform object detection on the image file
        img = Image.open(img_file)
        boxes = detect_objects_on_image(img)
        
        # label_file have ground truth  class, x_center, y_center, w, h
        temp_label_path = lbl_file.filename
        lbl_file.save(temp_label_path)
        labels = load_label_files(temp_label_path)
        os.remove(temp_label_path)
        
        print("labels: ", labels[0])
        print("boxes: ", boxes[0])
        
        all_boxes.append({
          "predictions": boxes,
          "labels": labels,
          # "mAP": mAP
        })
    
    return jsonify(all_boxes)

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
    global model
    
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

def detect_objects_on_image_YOLO(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format 
    [[x_center, y_center, w, h, object_type, probability],..]
    """
    global model
    
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        # Convert tensors to list
        xyxy = box.xyxy[0].tolist()
        cls = box.cls[0].item()
        conf = round(box.conf[0].item(), 2)
        
        # Calculate center coordinates
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = (xyxy[1] + xyxy[3]) / 2
        # Calculate width and height
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        
        output.append([
            x_center, y_center, w, h, result.names[cls], conf
        ])
    return output

set_up()
serve(app, host='0.0.0.0', port=8080)