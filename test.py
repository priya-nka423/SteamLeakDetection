from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train-5/weights/best.pt")

# Image path
image_path = "test.jpg"

# Run prediction
results = model.predict(
    source=image_path,
    conf=0.5,
    save=True,
    show=True
)

print("Prediction completed!")