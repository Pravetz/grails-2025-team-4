from app_preprocessing import preprocess_image
import re
from keras.models import load_model
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
import pandas as pd

TYPE_CASCADE = 'cascade'
TYPE_CUSTOM = 'custom'

def load_cnn_cascade(section):
	return {int(section[i]): load_model(section[i + 1]) for i in range(0, len(section), 2)}

def make_class_map(section):
	return {int(section[i]): int(section[i + 1]) for i in range(0, len(section), 2)}

def default_predict(model, preproc, data, image_size):
	processed_fragment = preproc(data, image_size)
	return {cls: model[cls].predict(processed_fragment)[0][0] for cls in model}

def execute_section(path, section):
	src = '\n'.join(section)
	result = {}
	exec(compile(src, path, "exec"), result)
	return result

def parse_model_cfg(path):
	cfg = ""
	with open(path, 'r', encoding='utf-8') as mcfg:
		cfg = mcfg.read()
	
	cfg_tokens = re.split(r'\n|\t', cfg)
	m_type = TYPE_CASCADE
	sections = {}
	
	i = 0
	while i < len(cfg_tokens):
		if not cfg_tokens[i]:
			i += 1
			continue
		else:
			match cfg_tokens[i]:
				case "type":
					i += 1
					m_type = cfg_tokens[i]
				case "section":
					i += 1
					section_name = cfg_tokens[i]
					sections[section_name] = []
					i += 1
					while cfg_tokens[i] != "endsection":
						sections[section_name].append(cfg_tokens[i])
						i += 1
		
		i += 1
	
	if m_type.lower() == TYPE_CASCADE:
		model = load_cnn_cascade(sections.get("classes", []))
		preprocessor = preprocess_image
		predict_func = default_predict
		classes = None
	else:
		temp = execute_section(path, sections.get("loader", []))
		model = temp["model"]
		temp = execute_section(path, sections.get("preproc", []))
		preprocessor = temp["preproc"]
		classes = make_class_map(sections.get("classes", []))
		temp = execute_section(path, sections.get("predict", []))
		predict_func = temp["predict"]
	
	return m_type.lower(), model, preprocessor, classes, predict_func

def remap_scores(scores, classes):
	if classes is None or not classes:
		return scores
	return {classes[cls]: scores[cls] for cls in scores}

class ModelInterface():
	def __init__(self, path):
		self.m_type, self.model, self.img_preprocessor, self.classes, self.predict_func = parse_model_cfg(path)
	
	def predict(self, yolo_results, image, label_mapping, image_size):
		valid_predictions = []
		height, width, _ = image.shape
		
		for box, predicted_class_id, confidence in yolo_results:
			box = box.astype(int)
			box[0] = max(0, box[0])
			box[1] = max(0, box[1])
			box[2] = min(width, box[2])
			box[3] = min(height, box[3])
		
			if box[2] <= box[0] or box[3] <= box[1]:
				continue
		
			yolo_predicted_class = label_mapping.get(predicted_class_id, None)
			if yolo_predicted_class is None:
				continue
		
			fragment = image[box[1]:box[3], box[0]:box[2]]
			if fragment.size == 0:
				continue
		
			scores = self.predict_func(self.model, self.img_preprocessor, fragment, image_size)
			scores = remap_scores(scores, self.classes)
			best_class, best_score = max(scores.items(), key=lambda x: x[1])
		
			sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
			if len(sorted_scores) > 1 and sorted_scores[0][1] == sorted_scores[1][1] or (sorted_scores[0][1] > 0.9 and sorted_scores[1][1] > 0.9):
				top_classes = [sorted_scores[0][0], sorted_scores[1][0]]
				if predicted_class_id in top_classes:
					best_class = predicted_class_id
				else:
					best_class = sorted_scores[0][0]
		
			if best_score > 0.5:
				valid_predictions.append((box, best_class, best_score))
		
		return valid_predictions