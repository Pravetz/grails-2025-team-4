from PySide6.QtWidgets import (
	QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
	QStackedWidget, QFileDialog, QTextEdit, QScrollArea, QSlider, QStyle, QListWidget, QListWidgetItem, QColorDialog
)
from PySide6.QtCore import Qt, QSize, QPointF, QRectF
from PySide6.QtGui import QPixmap, QImage, QColor, QKeyEvent
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
import json
import pickle
import copy

import app_info
import app_defaults
import app_inference
import app_utils
from app_models import ModelInterface

def cv2_to_pixmap(cv_image):
	height, width, channels = cv_image.shape
	q_image = QImage(cv_image.data, width, height, 3 * width, QImage.Format_RGB888)
	
	pixmap = QPixmap.fromImage(q_image)
	
	return pixmap


def dump_localization_en(path):
	with open(os.path.join("appdata", "localization", "en.json"), 'w') as udf:
		json.dump(
			app_defaults.LOCALIZATION_EN, 
			fp=udf, 
			indent=4
		)

def dump_localization_uk(path):
	with open(os.path.join("appdata", "localization", "uk.json"), 'w') as udf:
		json.dump(
			app_defaults.LOCALIZATION_UK, 
			fp=udf, 
			indent=4
		)

def create_appdata():
	os.makedirs("appdata", exist_ok=True)
	os.makedirs(os.path.join("appdata", "localization"), exist_ok=True)
	os.makedirs(os.path.join("appdata", "gcache"), exist_ok=True)
	with open(os.path.join("appdata", "userdat.json"), 'w') as udf:
		json.dump(
			app_defaults.USER_SETTINGS, 
			fp=udf, 
			indent=4
		)
	dump_localization_en(os.path.join("appdata", "localization", "en.json"))
	dump_localization_uk(os.path.join("appdata", "localization", "uk.json"))

def update_userdata(userdata):
	with open(os.path.join("appdata", "userdat.json"), 'w') as udf:
		json.dump(
			userdata, 
			fp=udf, 
			indent=4
		)

def check_integrity(userdata):
	for k, v in app_defaults.USER_SETTINGS.items():
		if k not in userdata:
			userdata[k] = v
	
	return userdata

def load_user_data():
	if not os.path.exists(os.path.join("appdata", "userdat.json")):
		create_appdata()
	
	with open(os.path.join("appdata", "userdat.json"), 'r') as udf:
		return check_integrity(json.load(udf))

def load_localization(path):
	with open(path, 'r') as lf:
		return json.load(lf)

def set_localized_text(loc_dict, key):
	return loc_dict.get(key, key)

def load_class_file(path):
	with open(path, 'r') as clsf:
		return {int(k) : v for k, v in json.load(clsf).items()}

class ImageLabel(QLabel):
	def __init__(self, label_text, parent_window, enable_file_io, parent=None):
		super().__init__(parent)
		self.p_window = parent_window
		self.setAlignment(Qt.AlignCenter)
		self.setStyleSheet("background-color: grey; border: 1px solid black;")
		self.setText(label_text)
		self.mousePressEvent = self.open_file_dialog if enable_file_io else None

		self.pixmap_original = None
		self.scale_factor = 1.0
		self.setFocusPolicy(Qt.StrongFocus)


	def open_file_dialog(self, event):
		file_dialog = QFileDialog(self)
		file_dialog.setNameFilters(["Images (*.png *.jpg *.jpeg *.bmp *.gif)"])
		if file_dialog.exec():
			if self.p_window.gallery_object_id[-1] is not None:
				app_utils.serialize_object(self.p_window.userdata_dict["GALLERY"][self.p_window.gallery_object_id[-1]], self.p_window.prediction[-1])
			self.p_window.gallery_object_id[-1] = None
			self.p_window.saved_gobj[-1] = False
			self.p_window.prediction[-1] = copy.deepcopy(app_defaults.GALLERY_OBJECT)
			self.p_window.image_path = file_dialog.selectedFiles()[0]
			self.pixmap_original = QPixmap(self.p_window.image_path)
			self.p_window.prediction[-1]["image"] = cv2.imread(self.p_window.image_path)
			self.p_window.prediction[-1]["image"] = cv2.cvtColor(self.p_window.prediction[-1]["image"], cv2.COLOR_BGR2RGB)
			self.update_image()

	def update_image(self):
		if self.pixmap_original:
			label_size = self.size()

			scaled_size = self.pixmap_original.size() * self.scale_factor
			scaled_pixmap = self.pixmap_original.scaled(
				scaled_size,
				Qt.KeepAspectRatio,
				Qt.SmoothTransformation
			)

			self.setPixmap(scaled_pixmap)

	def resizeEvent(self, event):
		super().resizeEvent(event)
		self.update_image()

	def keyPressEvent(self, event: QKeyEvent):
		if event.key() == Qt.Key_Plus:
			self.scale_factor *= 1.1
		elif event.key() == Qt.Key_Minus:
			self.scale_factor /= 1.1
		elif event.key() == Qt.Key_0:
			self.scale_factor = 1.0
		
		self.scale_factor = max(0.1, min(10.0, self.scale_factor))
		
		self.update_image()


class MainWindow(QWidget):
	def __init__(self, userdata_dict):
		super().__init__()
		self.userdata_dict = userdata_dict
		self.loclang = ""
		self.localization = {}
		
		self.shared_data = {
			"image_path" : str()
		}
		self.yolo_model = None
		self.frag_class = None
		self.prediction = [copy.deepcopy(app_defaults.GALLERY_OBJECT)]
		self.label_mapping = {}
		
		self.gallery_object_id = [None]
		self.img_scale_factor = [1.0]
		
		self.saved_gobj = [False]
		self.is_in_gallery_view = False
		self.class_color_click_signal_not_connected = True
		
		try:
			self.loclang = os.path.basename(userdata_dict["LOC_PATH"])
			self.loclang = self.loclang[: self.loclang.rfind('.')]
			self.localization = load_localization(userdata_dict["LOC_PATH"])
		except Exception:
			pass
		
		self.setWindowTitle(f"Construction Waste Detection[{set_localized_text(self.localization, 'recog_btn_text')}]")
		self.init_data()
		self.init_ui()

	def load_labels(self, override_colors, update_color_list=False):
		if self.userdata_dict["CLASS_PATH"]:
			class_data = load_class_file(self.userdata_dict["CLASS_PATH"])
			self.label_mapping = { k: v.get(self.loclang, v.get("default", f"Class_{k}")) for k, v in class_data.items() }
			self.userdata_dict["CLASS_COLORS"] = { self.label_mapping[k] : [randint(0,255),randint(0,255),randint(0,255)] if self.label_mapping[k] not in self.userdata_dict["CLASS_COLORS"] or override_colors else self.userdata_dict["CLASS_COLORS"][self.label_mapping[k]] for k, _ in class_data.items() }
			if update_color_list:
				self.populate_class_color_list()
	
	def init_data(self):
		if self.userdata_dict["YOLO_PATH"]:
			self.yolo_model = YOLO(self.userdata_dict["YOLO_PATH"])
		
		if self.userdata_dict["CNNC_PATH"]:
			self.frag_class = ModelInterface(self.userdata_dict["CNNC_PATH"])
		
		self.load_labels(override_colors=False)
	
	def init_ui(self):
		main_layout = QVBoxLayout()
	
		top_menu_layout = QGridLayout()
		self.recognition_button = QPushButton(set_localized_text(self.localization, "recog_btn_text"))
		self.gallery_button = QPushButton(set_localized_text(self.localization, "gallery_btn_text"))
		self.settings_button = QPushButton(set_localized_text(self.localization, "settings_btn_text"))
	
		top_menu_layout.addWidget(self.recognition_button, 0, 0)
		top_menu_layout.addWidget(self.gallery_button, 0, 1)
		top_menu_layout.addWidget(self.settings_button, 0, 2)
	
		self.stacked_widget = QStackedWidget()
	
		self.recognition_page = self.create_recognition_page()
		self.gallery_page = self.create_gallery_page()
		self.gallery_view_page = self.create_gallery_view_page()
		self.settings_page = self.create_settings_page()
	
		self.stacked_widget.addWidget(self.recognition_page)
		self.stacked_widget.addWidget(self.gallery_page)
		self.stacked_widget.addWidget(self.gallery_view_page)
		self.stacked_widget.addWidget(self.settings_page)
	
		bottom_info_layout = QVBoxLayout()
		self.version_label = QLabel(f"v{app_info.VERSION}, (c) {', '.join(app_info.AUTHORS)}")
		self.version_label.setAlignment(Qt.AlignLeft)
		self.version_label.setStyleSheet("border: 1px solid white;")
		bottom_info_layout.addWidget(self.version_label)
	
		self.recognition_button.clicked.connect(self.show_recog_page)
		self.gallery_button.clicked.connect(self.show_gallery)
		self.settings_button.clicked.connect(self.show_settings)
	
		main_layout.addLayout(top_menu_layout)
		main_layout.addWidget(self.stacked_widget, 1)
		main_layout.addLayout(bottom_info_layout)
		self.setLayout(main_layout)
	
		self.apply_scaling()

	def set_statistics_text(self, target):
		loc_text = set_localized_text(self.localization, "statistics_text")
		if loc_text == "statistics_text":
			target.setText(loc_text)
			return
		subs_table = app_defaults.STATISTICS_FMTK
		subs_table["IWIDTH"] = self.prediction[-1]["image"].shape[1] if self.prediction[-1]["image"] is not None else None
		subs_table["IHEIGHT"] = self.prediction[-1]["image"].shape[0] if self.prediction[-1]["image"] is not None else None
		if self.prediction[-1]["predictions"] is not None:
			subs_table["DCOUNT"] = len(self.prediction[-1]["predictions"])
			if self.prediction[-1]["classified"]:
				confidences = [c for _, _, c in self.prediction[-1]["predictions"]]
				object_list = [k for _, k, _ in self.prediction[-1]["predictions"]]
				object_counts = {}
				for o in object_list:
					if self.label_mapping[o] not in object_counts:
						object_counts[self.label_mapping[o]] = 1
						continue
					object_counts[self.label_mapping[o]] += 1
				subs_table["AVGC"] = round((sum(confidences) / len(confidences)) * 100.0, 2) if confidences else None
				subs_table["OBJC"] = app_utils.make_object_count_string(object_counts)
		loc_text = app_utils.format_text(loc_text, subs_table)
		target.setText(loc_text)

	def show_recog_page(self):
		if self.is_in_gallery_view:
			self.is_in_gallery_view = False
			self.prediction.pop()
			self.gallery_object_id.pop()
			self.saved_gobj.pop()
		if self.gallery_object_id[-1] is None:
			self.info_textbox.clear()
		self.setWindowTitle(f"Construction Waste Detection[{set_localized_text(self.localization, 'recog_btn_text')}]")
		self.stacked_widget.setCurrentWidget(self.recognition_page)
	
	def show_gallery(self):
		if self.is_in_gallery_view:
			self.is_in_gallery_view = False
			self.prediction.pop()
			self.gallery_object_id.pop()
			self.saved_gobj.pop()
		if not self.saved_gobj[-1] and self.gallery_object_id[-1] is not None:
			self.saved_gobj[-1] = True
			app_utils.serialize_object(self.userdata_dict["GALLERY"][self.gallery_object_id[-1]], self.prediction[-1])
		self.setWindowTitle(f"Construction Waste Detection[{set_localized_text(self.localization, 'gallery_btn_text')}]")
		self.update_gallery()
		self.stacked_widget.setCurrentWidget(self.gallery_page)
	
	def show_settings(self):
		if self.is_in_gallery_view:
			self.is_in_gallery_view = False
			self.prediction.pop()
			self.gallery_object_id.pop()
			self.saved_gobj.pop()
		if not self.saved_gobj[-1] and self.gallery_object_id[-1] is not None:
			self.saved_gobj[-1] = True
			app_utils.serialize_object(self.userdata_dict["GALLERY"][self.gallery_object_id[-1]], self.prediction[-1])
		self.setWindowTitle(f"Construction Waste Detection[{set_localized_text(self.localization, 'settings_btn_text')}]")
		self.populate_class_color_list()
		self.stacked_widget.setCurrentWidget(self.settings_page)

	def create_recognition_page(self):
		recognition_layout = QHBoxLayout()
	
		left_panel = QVBoxLayout()
		self.find_objects_button = QPushButton(set_localized_text(self.localization, "find_objs_btn_text"))
		self.classify_button = QPushButton(set_localized_text(self.localization, "classify_btn_text"))
		self.save_fragments_button = QPushButton(set_localized_text(self.localization, "savefrags_btn_text"))
		self.delete_gallery_button = QPushButton(set_localized_text(self.localization, "delfromgal_btn_text"))
		
		self.find_objects_button.clicked.connect(self.on_find_objects_button_click)
		self.classify_button.clicked.connect(self.on_classify_button_click)
		self.save_fragments_button.clicked.connect(self.on_save_fragments_button_click)
		self.delete_gallery_button.clicked.connect(self.delete_gallery_button_click)
	
		scroll_area = QScrollArea()
		self.info_textbox = QTextEdit()
		self.info_textbox.setReadOnly(True)
		scroll_area.setWidgetResizable(True)
		scroll_area.setWidget(self.info_textbox)
	
		left_panel.addWidget(self.find_objects_button)
		left_panel.addWidget(self.classify_button)
		left_panel.addWidget(self.save_fragments_button)
		left_panel.addWidget(self.delete_gallery_button)
		left_panel.addWidget(scroll_area, 1)
	
		image_scroll_area = QScrollArea()
		image_scroll_area.setWidgetResizable(True)
		self.image_label = ImageLabel(set_localized_text(self.localization, "choose_image_text"), self, True)
		image_scroll_area.setWidget(self.image_label);
	
		recognition_layout.addLayout(left_panel)
		recognition_layout.addWidget(image_scroll_area, 1)
	
		recognition_page = QWidget()
		recognition_page.setLayout(recognition_layout)
		return recognition_page

	def update_gallery(self):
		wscale = int(self.userdata_dict["QT_SCALE"] * 300.0)
		for i in reversed(range(self.gallery_layout.count())): 
			widget = self.gallery_layout.itemAt(i).widget()
			if widget is not None:
				widget.deleteLater()
	
		for i, gallery_obj in enumerate(self.userdata_dict["GALLERY"]):
			image_label = QLabel()
			prevpixmap = QPixmap(gallery_obj["preview_path"])
			image_label.setPixmap(prevpixmap.scaledToWidth(wscale))
			image_label.setAlignment(Qt.AlignCenter)
			image_label.setStyleSheet("border: 1px solid black; margin: 5px;")
			
			image_label.mousePressEvent = lambda event, idx=i, classified=gallery_obj["classified"], preview=prevpixmap, image_path=gallery_obj["image_path"], preds_path=gallery_obj["preds_path"]: self.show_gobj_view(idx, classified, preview, image_path, preds_path)
			
			self.gallery_layout.addWidget(image_label)
	
	def show_gobj_view(self, idx, classified, preview, image_path, preds_path):
		self.is_in_gallery_view = True
		self.gallery_object_id.append(idx)
		self.saved_gobj.append(True)
		self.gvp_image_label.pixmap_original = preview
		self.prediction.append(copy.deepcopy(app_defaults.GALLERY_OBJECT))
		self.prediction[-1]["classified"] = classified
		self.prediction[-1]["preview"] = preview
		self.prediction[-1]["image"] = app_utils.load_cv2_image_rgb(image_path)
		self.prediction[-1]["predictions"] = app_utils.deserialize_pickle(preds_path)
		self.set_statistics_text(self.gvp_info_textbox)
		self.gvp_image_label.update_image()
		self.stacked_widget.setCurrentWidget(self.gallery_view_page)
	
	def create_gallery_page(self):
		self.gallery_layout = QHBoxLayout()
		
		scroll_area = QScrollArea()
		scroll_area.setWidgetResizable(True)
		
		self.gallery_widget = QWidget()
		self.gallery_widget.setLayout(self.gallery_layout)
		
		scroll_area.setWidget(self.gallery_widget)
		
		gallery_page = QWidget()
		gallery_page_layout = QVBoxLayout()
		gallery_page_layout.addWidget(scroll_area)
		gallery_page.setLayout(gallery_page_layout)
		
		return gallery_page
	
	def create_gallery_view_page(self):
		gallery_view_layout = QHBoxLayout()
	
		self.gvp_save_fragments_button = QPushButton(set_localized_text(self.localization, "savefrags_btn_text"))
		self.gvp_delete_gallery_button = QPushButton(set_localized_text(self.localization, "delfromgal_btn_text"))
	
		self.gvp_save_fragments_button.clicked.connect(self.on_save_fragments_button_click)
		self.gvp_delete_gallery_button.clicked.connect(self.delete_gallery_button_click)
	
		left_panel = QVBoxLayout()
	
		scroll_area = QScrollArea()
		self.gvp_info_textbox = QTextEdit()
		self.gvp_info_textbox.setReadOnly(True)
		scroll_area.setWidgetResizable(True)
		scroll_area.setWidget(self.gvp_info_textbox)
		
		left_panel.addWidget(self.gvp_save_fragments_button)
		left_panel.addWidget(self.gvp_delete_gallery_button)
		left_panel.addWidget(scroll_area, 1)
		
		image_scroll_area = QScrollArea()
		image_scroll_area.setWidgetResizable(True)
		self.gvp_image_label = ImageLabel("", self, False)
		image_scroll_area.setWidget(self.gvp_image_label)
	
		gallery_view_layout.addLayout(left_panel)
		gallery_view_layout.addWidget(image_scroll_area, 1)
	
		gallery_view_page = QWidget()
		gallery_view_page.setLayout(gallery_view_layout)
		return gallery_view_page
	
	def create_settings_page(self):
		settings_layout = QVBoxLayout()
	
		scroll_area = QScrollArea()
	
		self.yolo_path_label = QLabel(set_localized_text(self.localization, "yolo_path_text"))
		self.yolo_path_field = QTextEdit(self.userdata_dict["YOLO_PATH"])
		self.yolo_path_field.setReadOnly(True)
		self.yolo_path_button = QPushButton()
		self.yolo_path_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
		self.yolo_path_button.clicked.connect(self.on_yolo_path_button_click)
		
		yolo_path_layout = QHBoxLayout()
		yolo_path_layout.addWidget(self.yolo_path_field)
		yolo_path_layout.addWidget(self.yolo_path_button)
	
		self.cnnc_path_label = QLabel(set_localized_text(self.localization, "cnnc_path_text"))
		self.cnnc_path_field = QTextEdit(self.userdata_dict["CNNC_PATH"])
		self.cnnc_path_field.setReadOnly(True)
		self.cnnc_path_button = QPushButton()
		self.cnnc_path_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
		self.cnnc_path_button.clicked.connect(self.on_cnnc_path_button_click)
		
		cnnc_path_layout = QHBoxLayout()
		cnnc_path_layout.addWidget(self.cnnc_path_field)
		cnnc_path_layout.addWidget(self.cnnc_path_button)
	
		self.class_path_label = QLabel(set_localized_text(self.localization, "class_path_text"))
		self.class_path_field = QTextEdit(self.userdata_dict["CLASS_PATH"])
		self.class_path_field.setReadOnly(True)
		self.class_path_button = QPushButton()
		self.class_path_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
		self.class_path_button.clicked.connect(self.on_class_path_button_click)
		
		class_path_layout = QHBoxLayout()
		class_path_layout.addWidget(self.class_path_field)
		class_path_layout.addWidget(self.class_path_button)
	
		self.lang_path_label = QLabel(set_localized_text(self.localization, "lang_path_text"))
		self.lang_path_field = QTextEdit(self.userdata_dict["LOC_PATH"])
		self.lang_path_field.setReadOnly(True)
		self.lang_path_button = QPushButton()
		self.lang_path_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
		self.lang_path_button.clicked.connect(lambda: self.open_file_dialog_cfg(self.lang_path_field, "Language Path", ["JSON (*.json)"], "LOC_PATH"))
		
		lang_path_layout = QHBoxLayout()
		lang_path_layout.addWidget(self.lang_path_field)
		lang_path_layout.addWidget(self.lang_path_button)
	
		self.imsize_label = QLabel(set_localized_text(self.localization, "imsize_text"))
		self.imsize_field = QTextEdit(f"{self.userdata_dict['IMAGE_SIZE'][0]}x{self.userdata_dict['IMAGE_SIZE'][1]}")
		self.imsize_field.textChanged.connect(self.on_imsize_text_changed)
		
	
		self.iou_label = QLabel(set_localized_text(self.localization, "iou_threshold_text"))
		self.iou_slider = QSlider(Qt.Horizontal)
		self.iou_slider.setMinimum(0)
		self.iou_slider.setMaximum(100)
		self.iou_slider.setValue(int(self.userdata_dict["IOU_THRESHOLD"] * 100))
		self.iou_slider.valueChanged.connect(self.on_iou_slider_value_changed)
	
		self.scale_label = QLabel(set_localized_text(self.localization, "ui_scale_text"))
		self.scale_slider = QSlider(Qt.Horizontal)
		self.scale_slider.setMinimum(25)
		self.scale_slider.setMaximum(200)
		self.scale_slider.setValue(int(self.userdata_dict["QT_SCALE"] * 100))
		self.scale_slider.valueChanged.connect(self.on_scale_slider_value_changed)
	
		self.font_size_label = QLabel(set_localized_text(self.localization, "font_size_text"))
		self.font_size_slider = QSlider(Qt.Horizontal)
		self.font_size_slider.setMinimum(8)
		self.font_size_slider.setMaximum(64)
		self.font_size_slider.setValue(self.userdata_dict["QT_FONT_SIZE"])
		self.font_size_slider.valueChanged.connect(self.on_font_size_slider_value_changed)
	
		self.cls_colors_label = QLabel(set_localized_text(self.localization, "cls_colors_text"))
		self.class_color_list = QListWidget()
	
		settings_layout.addWidget(self.yolo_path_label)
		settings_layout.addLayout(yolo_path_layout)
		settings_layout.addWidget(self.cnnc_path_label)
		settings_layout.addLayout(cnnc_path_layout)
		settings_layout.addWidget(self.class_path_label)
		settings_layout.addLayout(class_path_layout)
		settings_layout.addWidget(self.cls_colors_label)
		settings_layout.addWidget(self.class_color_list)
		settings_layout.addWidget(self.lang_path_label)
		settings_layout.addLayout(lang_path_layout)
		settings_layout.addWidget(self.imsize_label)
		settings_layout.addWidget(self.imsize_field)
		settings_layout.addWidget(self.iou_label)
		settings_layout.addWidget(self.iou_slider)
		settings_layout.addWidget(self.scale_label)
		settings_layout.addWidget(self.scale_slider)
		settings_layout.addWidget(self.font_size_label)
		settings_layout.addWidget(self.font_size_slider)
	
		self.clear_gallery_button = QPushButton(set_localized_text(self.localization, "clrgal_btn_text"))
		self.reset_settings_button = QPushButton(set_localized_text(self.localization, "resets_btn_text"))
		settings_layout.addWidget(self.clear_gallery_button)
		settings_layout.addWidget(self.reset_settings_button)
		
		self.clear_gallery_button.clicked.connect(self.on_clear_gallery_button_click)
		self.reset_settings_button.clicked.connect(self.on_reset_settings_button_click)
	
		settings_page = QWidget()
		settings_page.setLayout(settings_layout)
		scroll_area.setWidgetResizable(True)
		scroll_area.setWidget(settings_page)
	
		return scroll_area
	
	def populate_class_color_list(self):
		self.class_color_list.clear()
		
		for class_name, color in self.userdata_dict["CLASS_COLORS"].items():
			item = QListWidgetItem(class_name)
			item.setBackground(QColor(*color))
			self.class_color_list.addItem(item)
		
		if self.class_color_click_signal_not_connected:
			self.class_color_click_signal_not_connected = False
			self.class_color_list.itemClicked.connect(self.on_class_color_item_clicked)
	
	def on_class_color_item_clicked(self, item):
		class_name = item.text()
		current_color = QColor(*self.userdata_dict["CLASS_COLORS"][class_name])
		
		color = QColorDialog.getColor(current_color, self.class_color_list, f"Select Color for {class_name}")
		if color.isValid():
			new_rgb = [color.red(), color.green(), color.blue()]
			self.userdata_dict["CLASS_COLORS"][class_name] = new_rgb
			item.setBackground(color)
	
	def on_imsize_text_changed(self):
		text = self.imsize_field.toPlainText()
		if text.rfind('\n') == -1:
			return
		imsize = app_utils.extract_imsize(text)
		if imsize is None:
			self.imsize_field.setText(f"{self.userdata_dict['IMAGE_SIZE'][0]}x{self.userdata_dict['IMAGE_SIZE'][1]}")
			return
		self.userdata_dict["IMAGE_SIZE"] = imsize
	
	def on_clear_gallery_button_click(self):
		self.userdata_dict["GALLERY"].clear()
		self.gallery_object_id[-1] = None
		self.saved_gobj[-1] = False
	
	def on_reset_settings_button_click(self):
		for k, v in self.userdata_dict.items():
			if k != "GALLERY" and k != "CLASS_PATH" and k != "CLASS_COLORS" and k != "YOLO_PATH" and k != "CNNC_PATH":
				self.userdata_dict[k] = app_defaults.USER_SETTINGS[k]
		
		self.apply_scaling()
	
	def open_file_dialog_cfg(self, target_field, dialog_title, filters, userdata_key):
		file_dialog = QFileDialog(self, dialog_title)
		file_dialog.setNameFilters(filters)
		if file_dialog.exec():
			selected_path = file_dialog.selectedFiles()[0]
			target_field.setText(selected_path)
			if userdata_key:
				self.userdata_dict[userdata_key] = selected_path
	
	def on_save_fragments_button_click(self):
		if self.prediction[-1]["image"] is None or self.prediction[-1]["predictions"] is None:
			return
		file_dialog = QFileDialog(self, "Save fragments...")
		file_dialog.setFileMode(QFileDialog.Directory)
		if file_dialog.exec():
			selected_path = file_dialog.selectedFiles()[0]
			app_utils.dump_fragments_to_directory(self.prediction[-1], self.label_mapping, selected_path)
	
	def delete_gallery_button_click(self):
		if self.gallery_object_id[-1] is None:
			return
		rm_id = self.gallery_object_id[-1]
		last_id = len(self.userdata_dict["GALLERY"]) - 1
		app_utils.swap_files(self.userdata_dict["GALLERY"][rm_id]["preview_path"], self.userdata_dict["GALLERY"][-1]["preview_path"])
		app_utils.swap_files(self.userdata_dict["GALLERY"][rm_id]["image_path"], self.userdata_dict["GALLERY"][-1]["image_path"])
		app_utils.swap_files(self.userdata_dict["GALLERY"][rm_id]["proc_path"], self.userdata_dict["GALLERY"][-1]["proc_path"])
		app_utils.swap_files(self.userdata_dict["GALLERY"][rm_id]["preds_path"], self.userdata_dict["GALLERY"][-1]["preds_path"])
		self.userdata_dict["GALLERY"].pop()
		self.gallery_object_id[1] = None
		self.gallery_object_id[0] = app_utils.adjust_idx(self.gallery_object_id[0], rm_id, last_id)
	
	def on_yolo_path_button_click(self):
		self.open_file_dialog_cfg(self.yolo_path_field, "YOLO Path", ["PyTorch model (*.pt *.pth)"], "YOLO_PATH")
		if self.userdata_dict["YOLO_PATH"]:
			self.yolo_model = YOLO(self.userdata_dict["YOLO_PATH"])
	
	def on_cnnc_path_button_click(self):
		self.open_file_dialog_cfg(self.cnnc_path_field, "CNN Classifiers Path", ["Config (*.mlc)"], "CNNC_PATH")
		if self.userdata_dict["CNNC_PATH"]:
			self.frag_class = ModelInterface(self.userdata_dict["CNNC_PATH"])
	
	def on_class_path_button_click(self):
		self.open_file_dialog_cfg(self.class_path_field, "Class Data Path", ["JSON (*.json)"], "CLASS_PATH")
		self.load_labels(override_colors=True, update_color_list=True)
	
	def on_find_objects_button_click(self):
		if self.prediction[-1]["classified"]:
			return
		
		if self.yolo_model is not None and self.image_path is not None and self.image_path:
			self.prediction[-1]["predictions"] = app_inference.find_objects(self.image_path, self.yolo_model, self.userdata_dict["IOU_THRESHOLD"])
		
		self.prediction[-1]["classified"] = False
		self.prediction[-1]["proc_image"] = self.postproc_predictions_on_image(self.prediction[-1])
		self.image_label.pixmap_original = cv2_to_pixmap(self.prediction[-1]["proc_image"])
		self.prediction[-1]["preview"] = self.image_label.pixmap_original
		if self.gallery_object_id[-1] is None:
			self.gallery_object_id[-1] = len(self.userdata_dict["GALLERY"])
			self.userdata_dict["GALLERY"].append(app_utils.serializable_gallery_object(self.gallery_object_id[-1]))
		self.image_label.update_image()
		self.set_statistics_text(self.info_textbox)
		self.saved_gobj[-1] = False
	
	def on_classify_button_click(self):
		if self.frag_class is not None and not self.prediction[-1]["classified"]:
			if self.prediction[-1]["predictions"] is None and self.yolo_model is not None:
				self.prediction[-1]["predictions"] = app_inference.find_objects(self.image_path, self.yolo_model, self.userdata_dict["IOU_THRESHOLD"])
			
			self.prediction[-1]["predictions"] = self.frag_class.predict(self.prediction[-1]["predictions"], self.prediction[-1]["image"], self.label_mapping, self.userdata_dict["IMAGE_SIZE"])
			self.saved_gobj[-1] = False
		
		self.prediction[-1]["classified"] = True
		self.prediction[-1]["proc_image"] = self.postproc_predictions_on_image(self.prediction[-1], no_labels=False)
		self.image_label.pixmap_original = cv2_to_pixmap(self.prediction[-1]["proc_image"])
		self.prediction[-1]["preview"] = self.image_label.pixmap_original
		if self.gallery_object_id[-1] is None:
			self.gallery_object_id[-1] = len(self.userdata_dict["GALLERY"])
			self.userdata_dict["GALLERY"].append(app_utils.serializable_gallery_object(self.gallery_object_id[-1]))
		self.image_label.update_image()
		self.set_statistics_text(self.info_textbox)
	
	def postproc_predictions_on_image(self, prediction, no_labels=True):
		prediction["proc_image"] = prediction["image"].copy()
		for box, cls, conf in prediction["predictions"]:
			color = tuple(self.userdata_dict["CLASS_COLORS"][self.label_mapping[cls]])
			cv2.rectangle(prediction["proc_image"], (box[0], box[1]), (box[2], box[3]), color, 2)
		
			if not no_labels:
				label = f"{self.label_mapping[cls]}: {conf:.2f}"
				label_x, label_y = app_utils.adjust_text_position(
					box[0], box[1] - 10, label,
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2,
					prediction["proc_image"].shape
				)
				cv2.putText(prediction["proc_image"], label, (label_x, label_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
		
		return prediction["proc_image"]
	
	def on_iou_slider_value_changed(self, value):
		self.userdata_dict["IOU_THRESHOLD"] = value / 100.0
	
	def on_scale_slider_value_changed(self, value):
		self.userdata_dict["QT_SCALE"] = value / 100.0
		self.apply_scaling()
	
	def on_font_size_slider_value_changed(self, value):
		self.userdata_dict["QT_FONT_SIZE"] = value
		self.apply_scaling()
	
	def apply_scaling(self):
		font_scale = int(self.userdata_dict["QT_SCALE"] * self.userdata_dict["QT_FONT_SIZE"])
		button_height = int(self.userdata_dict["QT_SCALE"] * self.userdata_dict["QT_BHEIGHT"])
	
		self.recognition_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px;")
		self.gallery_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px;")
		self.settings_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px;")
	
		self.version_label.setStyleSheet(f"font-size: {font_scale}px;")
	
		self.find_objects_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px;")
		self.classify_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px;")
		self.save_fragments_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px;")
		self.delete_gallery_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px;")
		self.gvp_save_fragments_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px;")
		self.gvp_delete_gallery_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px;")
		self.info_textbox.setStyleSheet(f"font-size: {font_scale}px;")
		self.gvp_info_textbox.setStyleSheet(f"font-size: {font_scale}px;")
		self.yolo_path_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px; width: {button_height}px;")
		self.cnnc_path_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px; width: {button_height}px;")
		self.class_path_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px; width: {button_height}px;")
		self.lang_path_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px; width: {button_height}px;")
	
		self.yolo_path_label.setStyleSheet(f"font-size: {font_scale}px;")
		self.yolo_path_field.setStyleSheet(f"font-size: {font_scale}px;")
		self.cnnc_path_label.setStyleSheet(f"font-size: {font_scale}px;")
		self.cnnc_path_field.setStyleSheet(f"font-size: {font_scale}px;")
		self.class_path_label.setStyleSheet(f"font-size: {font_scale}px;")
		self.class_path_field.setStyleSheet(f"font-size: {font_scale}px;")
		self.cls_colors_label.setStyleSheet(f"font-size: {font_scale}px;")
		self.class_color_list.setStyleSheet(f"font-size: {font_scale}px;")
		self.lang_path_label.setStyleSheet(f"font-size: {font_scale}px;")
		self.lang_path_field.setStyleSheet(f"font-size: {font_scale}px;")
		self.imsize_label.setStyleSheet(f"font-size: {font_scale}px;")
		self.imsize_field.setStyleSheet(f"font-size: {font_scale}px;")
		self.iou_label.setStyleSheet(f"font-size: {font_scale}px;")
		self.scale_label.setStyleSheet(f"font-size: {font_scale}px;")
		self.font_size_label.setStyleSheet(f"font-size: {font_scale}px;")
		self.clear_gallery_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px;")
		self.reset_settings_button.setStyleSheet(f"font-size: {font_scale}px; height: {button_height}px;")
	
	def open_file_dialog(self, event):
		file_dialog = QFileDialog(self)
		file_dialog.setNameFilters(["Images (*.png *.jpg *.jpeg *.bmp *.gif)"])
		if file_dialog.exec():
			if self.gallery_object_id[-1] is not None:
				app_utils.serialize_object(self.userdata_dict["GALLERY"][self.gallery_object_id[-1]], self.prediction[-1])
			self.gallery_object_id[-1] = None
			self.saved_gobj[-1] = False
			self.prediction[-1] = copy.deepcopy(app_defaults.GALLERY_OBJECT)
			self.image_path = file_dialog.selectedFiles()[0]
			self.image_label.pixmap_original = QPixmap(self.image_path)
			self.prediction[-1]["image"] = cv2.imread(self.image_path)
			self.prediction[-1]["image"] = cv2.cvtColor(self.prediction[-1]["image"], cv2.COLOR_BGR2RGB)
			self.image_label.update_image()
	
	def resizeEvent(self, event):
		super().resizeEvent(event)
		self.image_label.update_image()
		self.gvp_image_label.update_image()
	
	def closeEvent(self, event):
		if self.gallery_object_id[-1] is not None:
			app_utils.serialize_object(self.userdata_dict["GALLERY"][self.gallery_object_id[-1]], self.prediction[-1])
		update_userdata(self.userdata_dict)
		super().closeEvent(event)


if __name__ == "__main__":
	userdata = load_user_data()
	app = QApplication([])

	window = MainWindow(userdata)
	window.resize(800, 600)
	window.show()

	app.exec()
