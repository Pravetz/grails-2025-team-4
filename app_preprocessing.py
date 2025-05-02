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

def preprocess_image(image, image_size):
	image = tf.image.resize_with_pad(image, image_size[0], image_size[1])
	image_array = tf.keras.preprocessing.image.img_to_array(image)
	image_array = np.expand_dims(image_array, axis=0)
	image_array = np.copy(image_array)
	image_array = tf.keras.applications.resnet.preprocess_input(image_array)
	
	return image_array
