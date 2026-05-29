from ultralytics import YOLO, RTDETR

# Load a COCO-pretrained YOLO26n model
model_1 = YOLO("yolo26m.pt")

# Load a COCO-pretrained RTDETR model
model_2 = RTDETR("rtdetr-l.pt")