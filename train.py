from ultralytics import YOLO

model = YOLO("yolo_models/yolo11m.pt")

model.train(data="data/dataset.yaml", imgsz=640, batch=8, epochs=25, workers=1, device=0)