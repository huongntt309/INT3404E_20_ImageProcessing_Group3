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

## Applications
### Sino-gradio
A Gradio interface for visualizing predictions from the fine-tuned YOLOv8 model.

### Sino-web
A web application for detecting Sino-nom characters in images, validating the val/test dataset through a user interface.

## Getting Started
 
Installation
Clone the repository:
```
git clone https://github.com/huongntt309/INT3404E_20_ImageProcessing_Group3.git
cd INT3404E_20_ImageProcessing_Group3
```
Usage
Run the web applications:

Sino-gradio:
```
cd sino-gradio
bash create_env.sh
bash run.sh
```
Sino-web-detector and Sino-web-val:
```
cd sino-web-service
bash create_env.sh
bash run.sh
```
Sino-model:

You can download notebook file at INT3404E_20_ImageProcessing_Group3/yolov8/YOLOv8.ipynb and upload them to Google Colab for usage.

## Contributors
1. Nguyen Thi Thuy Huong
2. Hoang Duc Anh
3. Tran Thi Van Anh
4. Tran Hanh Uyen
We welcome contributions and suggestions. Please feel free to open an issue or submit a pull request.
