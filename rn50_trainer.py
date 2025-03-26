import os
import shutil
import sys
if sys.argv.count("-nogpu") % 2 != 0:
	print("GPU was turned off")
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers, models
from keras.applications.resnet import preprocess_input
from tensorflow.keras.utils import img_to_array, array_to_img
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
import time
from PIL import Image, ImageEnhance

balance = True
epochs = 5
batch_size = 32
image_size = (224, 224)
dataset_dir = None
output_dir = None
target_class = None
model_name = "model"

def evaluate_model(model, val_dataset):
	print("Evaluating model...")
	y_pred = []
	y_val = []
	for x_batch, y_batch in val_dataset:
		preds = model.predict(x_batch)
		y_pred.extend((preds > 0.5).astype(int).flatten())
		y_val.extend(y_batch)
	
	y_pred = np.array(y_pred)
	y_val = np.array(y_val)
	
	precision = precision_score(y_val, y_pred, average="binary")
	recall = recall_score(y_val, y_pred, average="binary")
	f1 = f1_score(y_val, y_pred, average="binary")
	accuracy = accuracy_score(y_val, y_pred)
	
	results = f"""Validation Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}"""
	
	print(results)
	print("\nClassification Report:")
	print(classification_report(y_val, y_pred, target_names=["Non-Target", "Target"]))
	return results

def create_cnn():
	base_model = tf.keras.applications.ResNet50(input_shape=(image_size[0], image_size[1], 3), include_top=False, weights='imagenet')
	base_model.trainable = False
	model = models.Sequential([
		base_model,
		layers.GlobalAveragePooling2D(),
		layers.Dense(128, activation='relu'),
		layers.Dense(1, activation='sigmoid')
	])
	
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model

def generate_augmented_samples(filepaths, labels, augment_dir, image_size=(224, 224)):
	print(f"Generating augmented samples in {augment_dir}...")
	if os.path.exists(augment_dir):
		shutil.rmtree(augment_dir)
	os.makedirs(augment_dir, exist_ok=True)
	
	augmented_filepaths = []
	augmented_labels = []
	
	for filepath, label in zip(filepaths, labels):
		img = Image.open(filepath).convert("RGB")
	
		img.thumbnail(image_size, Image.BICUBIC)
		padded_img = Image.new("RGB", image_size, (0, 0, 0))
		offset = ((image_size[0] - img.size[0]) // 2, (image_size[1] - img.size[1]) // 2)
		padded_img.paste(img, offset)
	
		for i in range(4):
			rotated_img = padded_img.rotate(45 * (i + 1), resample=Image.BICUBIC)
	
			width, height = rotated_img.size
			left = (width - image_size[0]) / 2
			top = (height - image_size[1]) / 2
			right = (width + image_size[0]) / 2
			bottom = (height + image_size[1]) / 2
			cropped_img = rotated_img.crop((left, top, right, bottom))
	
			enhancer_brightness = ImageEnhance.Brightness(cropped_img)
			cropped_img = enhancer_brightness.enhance(np.random.uniform(0.8, 1.2))
	
			enhancer_contrast = ImageEnhance.Contrast(cropped_img)
			cropped_img = enhancer_contrast.enhance(np.random.uniform(0.8, 1.2))
	
			augmented_img_path = os.path.join(augment_dir, f"aug_{os.path.basename(filepath)}_{i}.jpg")
			cropped_img.save(augmented_img_path)
	
			augmented_filepaths.append(augmented_img_path)
			augmented_labels.append(label)
	
	return np.array(augmented_filepaths), np.array(augmented_labels)


def plot_training_history(path, history):
	plt.figure(figsize=(12, 4))
	
	plt.subplot(1, 2, 1)
	plt.plot(history.history['accuracy'], label='Train')
	plt.plot(history.history['val_accuracy'], label='Validation')
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(loc='upper left')
	
	plt.subplot(1, 2, 2)
	plt.plot(history.history['loss'], label='Train')
	plt.plot(history.history['val_loss'], label='Validation')
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(loc='upper left')
	
	plt.tight_layout()
	plt.savefig(path, dpi=300)

def load_data(dataset_dir, target_class):
	filepaths = []
	labels = []
	
	target_dir = os.path.join(dataset_dir, target_class)
	target_class_scount = len(os.listdir(os.path.join(dataset_dir, target_class))) if os.path.exists(target_dir) and os.path.isdir(target_dir) else 0
	if target_class_scount == 0:
		print(f"Target class is not found in {dataset_dir}")
		exit(1)
	balancing_count = int(round(target_class_scount / (len(os.listdir(dataset_dir)) - 1)))
	print(f"target class sample count is {target_class_scount}\ncount of samples for each class to balance is {balancing_count}")
	
	for class_name in os.listdir(dataset_dir):
		class_path = os.path.join(dataset_dir, class_name)
		if not os.path.isdir(class_path):
			continue
		label = int(class_name == target_class)
		
		bc = 0
		for img_name in os.listdir(class_path):
			if bc >= balancing_count and class_name != target_class and balance:
				break
			filepaths.append(os.path.join(class_path, img_name))
			labels.append(label)
			bc += 1
	
	values, counts = np.unique(labels, return_counts=True)
	print(values)
	print(counts)
	
	filepaths = np.array(filepaths)
	labels = np.array(labels)
	return filepaths, labels

def preprocess_image(filepath, label, augment=False):
	img = tf.io.read_file(filepath)
	img = tf.image.decode_image(img, channels=3)
	img = tf.image.resize_with_pad(img, image_size[0], image_size[1])
	
	if augment:
		img = tf.image.random_flip_left_right(img)
		img = tf.image.random_brightness(img, 0.2)
		img = tf.image.random_contrast(img, 0.8, 1.2)
		img = tf.image.random_crop(tf.image.resize(img, (256, 256)), (224, 224, 3))
	
	img = preprocess_input(img)
	return img, label

def create_dataset(filepaths, labels, augment=False):
	dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
	dataset = dataset.shuffle(buffer_size=len(filepaths))
	dataset = dataset.map(lambda x, y: preprocess_image(x, y, augment), num_parallel_calls=tf.data.AUTOTUNE)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)
	return dataset

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"""
Usage: {sys.argv[0]} <parameters>
Possible parameters:
	-data <path> = specify path to dataset
	-o <path> = output path (must be a directory)
	-epochs <count> = set epoch count (default is 5)
	-target <class_name> = set target class (based on classes in input data)
	-isz <WxH> = set image size for model
	-nogpu = don't use GPU for training/testing, does nothing on systems with no GPU
		""")
		exit()
	
	for i in range(1, len(sys.argv), 2):
		if sys.argv[i] == "-data":
			dataset_dir = sys.argv[i + 1]
		elif sys.argv[i] == "-o":
			output_dir = sys.argv[i + 1]
		elif sys.argv[i] == "-epochs":
			epochs = int(sys.argv[i + 1])
		elif sys.argv[i] == "-target":
			target_class = sys.argv[i + 1]
		elif sys.argv[i] == "-isz":
			try:
				sz_tokens = sys.argv[i + 1].lower().split('x')
				image_size = (int(sz_tokens[0]), int(sz_tokens[1]))
			except:
				image_size = (224, 224)
		elif sys.argv[i] == "-bal":
			balance = not balance
	
	if dataset_dir is None or output_dir is None:
		print("Path to dataset or output directory is not specified.")
		exit()
	
	if not os.path.exists(output_dir):
		os.makedirs(output_dir, exist_ok=True)
	
	print("Loading data...")
	filepaths, labels = load_data(dataset_dir, target_class)
	
	augment_dir = os.path.join(output_dir, "augmented_samples")
	augmented_filepaths, augmented_labels = generate_augmented_samples(filepaths, labels, augment_dir)
	
	combined_filepaths = np.concatenate([filepaths, augmented_filepaths])
	combined_labels = np.concatenate([labels, augmented_labels])
	
	X_train, X_val, y_train, y_val = train_test_split(combined_filepaths, combined_labels, test_size=0.2, random_state=42)
	
	train_dataset = create_dataset(X_train, y_train, augment=False)
	val_dataset = create_dataset(X_val, y_val, augment=False)
	
	class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
	class_weights = {i: class_weights[i] for i in range(len(class_weights))}
	
	print("Class weights:", class_weights)
	
	model = create_cnn()
	
	print("Training model...")
	history = model.fit(
		train_dataset,
		validation_data=val_dataset,
		epochs=epochs,
		class_weight=class_weights
	)
	
	model_path = os.path.join(output_dir, f"{model_name}.keras")
	model.save(model_path)
	print(f"Model saved to {model_path}")
	
	history_path = os.path.join(output_dir, "training_history.png")
	plot_training_history(history_path, history)
	print(f"Training history plot saved to {history_path}")
	results = evaluate_model(model, val_dataset)
	with open(os.path.join(output_dir, "eval.txt"), 'w', encoding='utf-8') as evf:
		evf.write(results)
