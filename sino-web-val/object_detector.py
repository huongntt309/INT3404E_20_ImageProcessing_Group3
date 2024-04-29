import os
from ultralytics import YOLO
from flask import jsonify, request, Response, Flask, template_rendered
from waitress import serve
from PIL import Image
import json
import yaml

app = Flask(__name__, static_url_path='/static')
model = None

def set_up():
  global model
  file_dir = os.path.dirname(os.path.realpath(__file__))
  model_path = os.path.join(file_dir, "best.pt")
  model = YOLO(model_path)
  
  print("Running at: http://localhost:8080")


def save_img_label(image_files, label_files):
    # Ensure the number of image files and label files match
    if len(image_files) != len(label_files):
        return jsonify({"error": "Number of image files and label files do not match"}), 400

    # Create a directory to save images if it doesn't exist
    file_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(file_dir, "images/val")
    label_dir = os.path.join(file_dir, "labels/val")  
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    # Save uploaded files
    for image_file, label_file in zip(image_files, label_files):
        # Save image file
        image_filename = os.path.join(save_dir, image_file.filename)
        image_file.save(image_filename)
        
        # Save label file
        label_filename = os.path.join(label_dir, label_file.filename)
        label_file.save(label_filename)
        
        print("Saving image file: " + label_filename)

def remove_img_label():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(file_dir, "images/val")
    label_dir = os.path.join(file_dir, "labels/val")  

    # Remove image files
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Remove label files
    for filename in os.listdir(label_dir):
        file_path = os.path.join(label_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    print("Image and label files removed successfully.")

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
    
    save_img_label(image_files, label_files)
    
    # Customize validation settings
    file_dir = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(file_dir, "custom_data.yaml")
    
    global model
    validation_results = model.val(data = yaml_path,
                                        imgsz=896,
                                        batch=16,
                                        conf=0.25,
                                        iou=0.6)
    
    remove_img_label()
    # Export results to frontend
    
    print("box_maps BE:", validation_results.box.maps)
    box_maps_value = validation_results.box.maps.item()
    result = {
        "box_map": validation_results.box.map,
        "box_map50": validation_results.box.map50,
        "box_map75": validation_results.box.map75,
        "box_maps": box_maps_value
    }
    print("result BE:", result)
    return jsonify(result)


set_up()
serve(app, host='0.0.0.0', port=8080)