from ultralytics import YOLO

model = YOLO("custom.pt")

model.predict(source="test.png", show=True, save=True)