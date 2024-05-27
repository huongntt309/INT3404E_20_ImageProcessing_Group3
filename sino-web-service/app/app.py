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
  
  print("Running at: http://localhost:8080/validate")


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

@app.route("/validate")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    file_dir = os.path.dirname(os.path.realpath(__file__))
    html_path = os.path.join(file_dir, "index_val.html")
    with open(html_path) as file:
        return file.read()

@app.route("/predict")
def predict_page():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    file_dir = os.path.dirname(os.path.realpath(__file__))
    html_path = os.path.join(file_dir, "index_predict.html")
    with open(html_path) as file:
        return file.read()
    
@app.route("/validate_api", methods=["POST"])
def validate_api():
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
    # Get configuration
    imageSize = int(request.form['imageSize'])
    iou = float(request.form['iou'])  # Assuming IOU is a floating-point number
    confidence = float(request.form['confidence'])  # Assuming confidence is a floating-point number

    
    
    save_img_label(image_files, label_files)
    
    # Customize validation settings
    file_dir = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(file_dir, "custom_data.yaml")
    
    global model
    validation_results = model.val(data = yaml_path,
                                        imgsz=imageSize,
                                        batch=16,
                                        conf=confidence,
                                        iou=iou)
    
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

@app.route("/predict_api", methods=["POST"])
def predict_api():
    """
        Handler of /predict POST endpoint
        Receives uploaded file with a name "image_file", 
        passes it through YOLOv8 object detection 
        network and returns an array of bounding boxes.
        :return: a JSON array of objects bounding 
        boxes in format 
        [[x1,y1,x2,y2,object_type,probability],..]
    """
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
    
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(
      json.dumps(boxes),  
      mimetype='application/json'
    )


set_up()
serve(app, host='0.0.0.0', port=8080)