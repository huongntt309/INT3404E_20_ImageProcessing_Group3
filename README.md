# Sino-nom Character Localization Project
This project aims to develop and implement a system for detecting and localizing Sino-nom characters in images using various state-of-the-art object detection models. Our final solution leverages the YOLOv8 model, given its superior performance on our dataset.

## Approaches
### Model Selection
We explored several object detection models for the Sino-nom Character Localization dataset:

1. Detectron2: A modular library for advanced object detection tasks.
2. DETR: End-to-end object detection using Transformers and CNNs.
3. YOLOv5: Fast and accurate real-time object detection system.
4. YOLOv8: Enhanced YOLO series model with improved detection accuracy.
5. YOLOv9: The latest YOLO model offering significant detection accuracy improvements.

Results: Using the Average Precision (AP) metric, we found that YOLOv8 outperformed other models with an mAP of approximately 0.8. Therefore, we chose YOLOv8 as our primary model.

### Data Augmentation
To enhance model performance, we applied various data augmentation techniques to address issues like small character size, white characters on a black background, and stained images. We utilized the Albumentations library, which offers a range of pre-defined and customizable transformations.

## Task Descriptions
1. Nguyen Thi Thuy Huong
  - Implementing a pipeline in Detectron2
  - Conducting error analysis
  - Data augmentation
  - Reviewing and refining documentation
2. Hoang Duc Anh

  - Implementing a pipeline in YOLOv9
  - Implementing and tuning YOLOv8 (V8L, V8M)
  - Training with augmented data
  - Evaluating and selecting data for augmentation
3. Tran Thi Van Anh

  - Implementing a pipeline in DETR
  - Sino-nom data analysis
  - Documenting project workflow and methodologies
  - Evaluating and selecting data for augmentation
4. Tran Hanh Uyen

  - Implementing a pipeline in YOLOv5
  - Gathering and curating datasets
  - Filtering augmented data for training
  - Self-Evaluation of Contributions
Each team member's contributions were essential in achieving our project's goals. The detailed tasks undertaken by each member ensured a thorough and collaborative approach to model development, data augmentation, and system implementation.

## Applications
### Sino-gradio
A Gradio interface for visualizing predictions from the fine-tuned YOLOv8 model.

### Sino-web
A web application for detecting Sino-nom characters in images, validating the val/test dataset through a user interface.

## Getting Started
Prerequisites
1. Python 3.10 +
2. PyTorch
3. YOLOv8
   
Installation
Clone the repository:
```
git clone https://github.com/huongntt309/INT3404E_20_ImageProcessing_Group3.git
cd INT3404E_20_ImageProcessing_Group3
```
Usage
Train the model:

```
python train.py --config configs/yolov8_config.yaml
```

Run the web applications:

Sino-gradio:
```
python sino-gradio/app.py
```
Sino-web-detector and Sino-web-val:
```
python sino_web_service/object_detector.py
```


## Contributors
1. Nguyen Thi Thuy Huong
2. Hoang Duc Anh
3. Tran Thi Van Anh
4. Tran Hanh Uyen
We welcome contributions and suggestions. Please feel free to open an issue or submit a pull request.
