# Realtime_Object_Detection
Real-time object detection through Laptop or PC Webcam/Video using Python+ Tensorflow + OpenCV

# Overview
This project is written in Python to perform real-time object detection on security cameras, webcams or video files.

The pre-trained model COCO SSD was used with Tensorflow to recognise objects in the video frame that was collected. OpenCV was then used to draw a rectangle where the object was found and to produce a text that included the name of the object that was detected.

 1. Tensorflow 2.10
 2. OpenCV
 3. Model COCO SSD
 4. Python 3.10

# Run

```
python Object_Dectection.py
```

# Demo

![Screenshot 2024-01-26 165046](https://github.com/xamaryadav/Realtime_Object_Detection/assets/93003722/44b4572b-aa53-4aec-831b-5a9e8d75269e)

# Setup

Step 1: Install Dependencies

```
pip install tensorflow="2.10"
```
```
pip install opencv-python numpy
```
```
pip install tensorflow
pip install tf_slim
pip install -U protobuf
pip install -U six
pip install -U lxml
pip install -U matplotlib
pip install -U Cython
pip install -U contextlib2
pip install -U pillow
pip install -U beautifulsoup4
pip install -U tf_slim

```
Step 2: Clone the TensorFlow Models Repository

```
git clone https://github.com/tensorflow/models.git
cd models/research
```
Download Protoc Compiler and add bin to path then run 
```
protoc object_detection/protos/*.proto --python_out=.
```
 Navigate to the models/research directory
```
cd C:\Users\Amar\AppData\Local\Programs\Python\Python310\models\research
```
 Set PYTHONPATH
```
set PYTHONPATH=%PYTHONPATH%;%cd%;%cd%\slim
```
Step 3 : Install COCO API

 Navigate to the models/research directory
```
cd models/research
```
 Clone the COCO API repository
```
git clone https://github.com/cocodataset/cocoapi.git
```
 Navigate to the cocoapi/PythonAPI directory
```
cd C:\Users\Amar\AppData\Local\Programs\Python\Python310\models\research\cocoapi\PythonAPI
```
 Copy the pycocotools directory
```
copy pycocotools C:\path\to\tensorflow\models\research
```
 Set Python Path
```
set PYTHONPATH=%PYTHONPATH%;C:\Users\Amar\AppData\Local\Programs\Python\Python310\models\research;C:\Users\Amar\AppData\Local\Programs\Python\Python310\models\research\slim
```






