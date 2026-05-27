# Steam Leak Detection using YOLOv8

An AI-powered industrial steam leak detection system developed using YOLOv8, OpenCV, and Python for real-time industrial safety monitoring.

## 📌 Project Overview

This project detects steam leak regions from industrial thermal/RGB images using a custom-trained YOLOv8 object detection model. The system helps improve industrial safety by automatically identifying possible steam leak locations in pipelines, boilers, and industrial equipment.

The model is trained on a custom annotated dataset and can perform real-time detection with bounding boxes and confidence scores.

---

## 🚀 Features

- Real-time steam leak detection
- YOLOv8 custom object detection model
- Bounding box visualization
- Confidence score prediction
- Thermal/RGB image support
- Custom dataset training
- Industrial safety monitoring application

---

## 🛠️ Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- PyTorch
- VS Code

---

## 📂 Project Structure

```bash
SteamLeakDetection/
│
├── combined_dataset/
├── runs/
├── train.py
├── test.py
├── popup_test.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset Information

- Custom steam leak detection dataset
- Annotated using YOLO format labels
- Dataset split:
  - Train
  - Validation
  - Test

The dataset contains industrial steam leak images captured under different environmental conditions.

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/priya-nka423/SteamLeakDetection.git
cd SteamLeakDetection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```
### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

Run the training script:

```bash
python train.py
```

The trained model weights will be saved inside:

```bash
runs/detect/train-5/weights/best.pt
```

---

## 🔍 Model Testing

Run detection/testing:

```bash
python test.py
```

or

```bash
python popup_test.py
```

---

## 📈 Output Results

The model generates:
- Bounding boxes
- Detection confidence scores
- Processed output images/videos

Output files are saved inside:

```bash
runs/detect/train-5
```

---

## 🖼️ Results 
Example:

```md
![Detection Result](results/sample_output.png)
```

---

## 🎯 Future Improvements

- IoT-based industrial monitoring
- Alarm/notification system
- Cloud monitoring dashboard

