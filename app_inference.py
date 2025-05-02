import os
import sys
from random import randint
from ultralytics import YOLO
import torch
import cv2
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from keras.models import load_model
from keras.applications.resnet import preprocess_input
import pandas as pd

def non_maximum_suppression(boxes, scores, iou_threshold=0.75):
	return cv2.dnn.NMSBoxes(
		bboxes=boxes,
		scores=scores,
		score_threshold=0.0,
		nms_threshold=iou_threshold
	)

def find_objects(image_path, yolo_model, iou_threshold):
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	if "cuda" in device:
		torch.cuda.set_device(0)
	
	results = yolo_model(image_path)
	
	processed_predictions = []
	
	for result in results:
		if result.boxes is None:
			continue
		
		boxes = result.boxes.xyxy.cpu().numpy() 
		confidences = result.boxes.conf.cpu().numpy() 
		classes = result.boxes.cls.cpu().numpy()
		
		keep_indices = non_maximum_suppression(boxes, confidences, iou_threshold)
		
		processed_predictions.extend([
			(boxes[i].astype(int), int(classes[i]), confidences[i]) for i in keep_indices
		])
	
	return processed_predictions