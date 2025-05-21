## YOLO for object detection
This branch contains simple and configurable Python script to train YOLO for object detection using Ultalytics library.
```
Usage: u_yolo_train.py <parameters>
Possible parameters:
	-yolov <string> = specify YOLO version, 'yolov8n' by default
	-m <string> = specify trained model file name
	-teston <path> = give image for testing the trained model
	-nogpu = don't use GPU for training
	-imgsz <S> = specify image size integer
	-epochs = specify epoch count to train model
	-val = turn validation mode on, off by default
```
