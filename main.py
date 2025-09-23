from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict(r"l29.png", save=True)
print(results)
