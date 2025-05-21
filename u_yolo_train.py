from ultralytics import YOLO
import torch
import sys
import yaml
from pathlib import Path

yolo_version = "yolov8n.pt"
model_name = "yolo_trained.pt"
test_image = ""
epochs = 5
imgsz = 640
nogpu = False

def calculate_iou(pred_box, gt_box):
	x1, y1, x2, y2 = pred_box
	gx1, gy1, gx2, gy2 = gt_box
	
	inter_x1 = max(x1, gx1)
	inter_y1 = max(y1, gy1)
	inter_x2 = min(x2, gx2)
	inter_y2 = min(y2, gy2)
	
	inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
	
	pred_area = (x2 - x1) * (y2 - y1)
	gt_area = (gx2 - gx1) * (gy2 - gy1)
	union_area = pred_area + gt_area - inter_area
	
	iou = inter_area / union_area if union_area > 0 else 0
	return iou

def compute_miou(model, data_path, img_width=1920, img_height=1200):
	with open(data_path, 'r') as f:
		data = yaml.safe_load(f)
	val_images_path = Path(data['path']) / data['val'] / "images"
	val_labels_path = Path(data['path']) / data['val'] / "labels"

	all_ious = []

	for image_path in val_images_path.glob('*.jpg'):
		label_path = val_labels_path / (image_path.stem + '.txt')

		with open(label_path, 'r') as f:
			ground_truths = []
			for line in f:
				_, x_center, y_center, width, height = map(float, line.strip().split())
				x1 = int((x_center - width / 2) * img_width)
				y1 = int((y_center - height / 2) * img_height)
				x2 = int((x_center + width / 2) * img_width)
				y2 = int((y_center + height / 2) * img_height)
				ground_truths.append((x1, y1, x2, y2))

		results = model(str(image_path))
		pred_boxes = results[0].boxes.xyxy.cpu().numpy()

		for pred_box in pred_boxes:
			best_iou = 0
			for gt_box in ground_truths:
				iou = calculate_iou(pred_box, gt_box)
				best_iou = max(best_iou, iou)
			all_ious.append(best_iou)

	miou = sum(all_ious) / len(all_ious) if all_ious else 0
	return miou



if __name__ == "__main__":
	validation_mode = False
	datapath = ""
	i = 1
	while i < len(sys.argv):
		if sys.argv[i].lower() == "-yolov" and i + 1 < len(sys.argv):
			yolo_version = sys.argv[i + 1]
			i += 1
		if sys.argv[i].lower() == "-m" and i + 1 < len(sys.argv):
			model_name = sys.argv[i + 1]
			i += 1
		if sys.argv[i].lower() == "-teston" and i + 1 < len(sys.argv):
			test_image = sys.argv[i + 1]
			i += 1
		if sys.argv[i].lower() == "-nogpu" and i + 1 < len(sys.argv):
			nogpu = not nogpu
		
		if sys.argv[i].lower() == "-imgsz" and i + 1 < len(sys.argv):
			try:
				imgsz = int(sys.argv[i + 1])
			except Exception as e:
				print(f"{e}, image size is reset to {imgsz}")
			
			i += 1
		
		if sys.argv[i].lower() == "-epochs" and i + 1 < len(sys.argv):
			try:
				epochs = int(sys.argv[i + 1])
			except Exception as e:
				print(f"{e}, epochs count is reset to {epochs}")
			
			i += 1
		if sys.argv[i].lower() == "-val":
			validation_mode = True
			if i + 1 < len(sys.argv) and sys.argv[i + 1][0] != '-':
				datapath = sys.argv[i + 1]
				i += 1
		
		i += 1
	
	device = "0" if torch.cuda.is_available() and not nogpu else "cpu"
	if device == "0":
		torch.cuda.set_device(0)
	
	model = YOLO(f"{yolo_version.lower()}.pt")
	
	if not validation_mode:
		train_results = model.train(
			data="data.yaml",
			epochs=epochs,
			imgsz=imgsz,
			device=device,
		)
	
	metrics = model.val() if not validation_mode or not datapath else model.val(data=datapath, imgsz=imgsz, device=device)
	if test_image:
		results = model(test_image)
		results[0].show()
	
	miou = compute_miou(model, "data.yaml")
	print(f"miou = {miou:.4f}")
	
	model.save(f"{model_name}.pt")