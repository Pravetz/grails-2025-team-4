import os
import uuid
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

from PySide6.QtGui import QPixmap, QImageWriter
from PySide6.QtCore import QBuffer, QByteArray

import json
import pickle
import re

def adjust_text_position(x, y, text, font, font_scale, thickness, image_shape):
	text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
	text_width, text_height = text_size[0], text_size[1]

	if y - text_height < 0:
		y = text_height + 5
	if x + text_width > image_shape[1]:
		x = image_shape[1] - text_width - 5

	return x, y
	

def serializable_gallery_object(gallery_obj_id):
	return {
		"classified" : False,
		"preview_path" : os.path.join("appdata", "gcache", f"o{hex(gallery_obj_id)}.preview.png"),
		"image_path" : os.path.join("appdata", "gcache", f"o{hex(gallery_obj_id)}.image.png"),
		"proc_path" : os.path.join("appdata", "gcache", f"o{hex(gallery_obj_id)}.proc.png"),
		"preds_path" : os.path.join("appdata", "gcache", f"o{hex(gallery_obj_id)}.preds")
	}

def nullify_idx(lst, idx):
	for i, x in enumerate(lst):
		if x == idx:
			lst[i] = None
	return lst

def adjust_idx(base, removed_id, last_id):
	if base is None:
		return None
	if base == last_id:
		return removed_id
	if base == removed_id:
		return None
	
	return base

def swap_files(fnl, fnr):
	if not os.path.exists(fnl) or not os.path.exists(fnr) or fnl == fnr:
		return
	
	tmp = "file.tmp"
	
	os.rename(fnl, tmp)
	os.rename(fnr, fnl)
	os.rename(tmp, fnr)

def save_qpixmap_as_png(qpixmap, path):
	pixmap_image = qpixmap.toImage()
	pixmap_image.save(path, "PNG")

def save_cv2_image_as_png(cv2_image, path):
	cv2.imwrite(path, cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR))

def deserialize_pickle(path):
	with open(path, 'rb') as pf:
		return pickle.load(pf)

def extract_imsize(string):
	tokens = re.split(r'x', string.lower().strip())
	
	if len(tokens) != 2:
		return None
	try:
		return [int(tokens[0]), int(tokens[1])]
	except Exception:
		return None

def make_object_count_string(objects):
	result = ""
	for k, v in objects.items():
		result += f"{k}: {v}\n"
	return result

def format_text(string, subs_table):
	def replace_placeholder(match):
		key = match.group(1)
		value = subs_table.get(key)
		return str(value) if value is not None else "---"
	
	return re.sub(r'\[([A-Z]+)\]', replace_placeholder, string)

def load_cv2_image_rgb(path):
	img = cv2.imread(path)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def serialize_object(serializable_gobj, gobj):
	serializable_gobj["classified"] = gobj["classified"]
	save_qpixmap_as_png(gobj["preview"], serializable_gobj["preview_path"])
	
	save_cv2_image_as_png(gobj["image"], serializable_gobj["image_path"])
	save_cv2_image_as_png(gobj["proc_image"], serializable_gobj["proc_path"])

	with open(serializable_gobj["preds_path"], 'wb') as pf:
		pickle.dump(gobj["predictions"], pf)

def dump_fragments_xml_annotation(image, predictions, label_mapping, path, image_name):
	iheight, iwidth, idepth = image.shape
	if not os.path.exists(path):
		os.makedirs(path)
	
	xml_filename = image_name + ".xml"
	with open(os.path.join(path, xml_filename), 'w', encoding='utf-8') as xmlf:
		xmlf.write(f"""<?xml version="1.0" encoding="utf-8"?>
<annotation>
	<folder/>
	<filename>{image_name}.png</filename>
	<path>{image_name}.png</path>
	<source></source>
	<size>
			<width>{iwidth}</width>
			<height>{iheight}</height>
			<depth>{idepth}</depth>
	</size>
	<segmented>0</segmented>""")
		for box, cls, conf in predictions:
			xmlf.write(f"""
	<object>
		<name>{label_mapping[cls]}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<bndbox>
			<xmin>{box[0]}</xmin>
			<xmax>{box[2]}</xmax>
			<ymin>{box[1]}</ymin>
			<ymax>{box[3]}</ymax>
		</bndbox>
		<polygon>
		</polygon>
	</object>""")
		xmlf.write("</annotation>")

def dump_fragments_yolo_annotation(image, predictions, label_mapping, path, image_name):
	iheight, iwidth, idepth = image.shape
	if not os.path.exists(path):
		os.makedirs(path)
	
	txt_filename = image_name + ".txt"
	
	with open(os.path.join(path, txt_filename), 'w', encoding='utf-8') as txt:
		for box, cls, conf in predictions:
			txt.write(f"{cls} {(box[0] + box[2]) / 2 / iwidth} {(box[1] + box[3]) / 2 / iheight} {(box[2] - box[0]) / iwidth} {(box[3] - box[1]) / iheight}\n")

def dump_fragments_to_directory(prediction, label_mapping, path):
	for box, cls, conf in prediction["predictions"]:
		class_path = os.path.join(path, label_mapping[cls]) if prediction["classified"] else os.path.join(path, "fragments")
		if not os.path.exists(class_path):
			os.makedirs(class_path)
		fragment = prediction["image"][box[1]:box[3], box[0]:box[2]]
		if fragment.size == 0:
			continue
	
		save_cv2_image_as_png(fragment, os.path.join(class_path, f"{uuid.uuid4().hex}.png"))
		
	if prediction["classified"]:
		image_name = f"{uuid.uuid4().hex}"
		save_cv2_image_as_png(prediction["image"], os.path.join(path, "annotated_images", image_name + ".png"))
		dump_fragments_xml_annotation(prediction["image"], prediction["predictions"], label_mapping, os.path.join(path, "annotated_images"), image_name)
		dump_fragments_yolo_annotation(prediction["image"], prediction["predictions"], label_mapping, os.path.join(path, "annotated_images"), image_name)
	