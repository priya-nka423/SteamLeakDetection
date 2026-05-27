from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="C:/Users/HP/Desktop/SteamLeakDetection/combined_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu"
    
)