from ultralytics import YOLO
import cv2
import os
from tkinter import Tk, filedialog
import sys
from pathlib import Path
from datetime import datetime

# Settings
args = {"no_display": False, "auto_close": 3}
project_root = Path(__file__).resolve().parent.parent
initial_dir = project_root / "combined_dataset" / "test" / "images"

# Image selection - GUI or Console (BEFORE loading model)
def select_gui():
    try:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.focus_force()
        path = filedialog.askopenfilename(
            title="Select test image",
            initialdir=str(initial_dir),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        root.destroy()
        return path
    except:
        return None

def select_console():
    candidates = sorted(list(initial_dir.glob("*.jpg")) + list(initial_dir.glob("*.jpeg")) + 
                       list(initial_dir.glob("*.png")) + list(initial_dir.glob("*.bmp")))
    if not candidates:
        return None
    for i, p in enumerate(candidates[:20], 1):
        print(f"  {i}. {p.name}")
    if len(candidates) > 20:
        print(f"  ...and {len(candidates)-20} more")
    choice = input("Enter number (or press Enter for first): ").strip()
    return str(candidates[0]) if not choice else str(candidates[int(choice)-1]) if choice.isdigit() else None

print("Select image:")
image_path = select_gui() or select_console()
if not image_path:
    sys.exit("No image selected")

# Load model AFTER image selection
print("Loading model...")
model = YOLO("runs/detect/train-4/weights/best.pt")
print("✓ Model loaded\n")

# Detection
print(f"Detecting: {Path(image_path).name}...")
results = model.predict(source=image_path, conf=0.25, save=False)
img_bgr = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
print(f"✓ Found {len(results[0].boxes)} object(s)\n")

# Save
save_dir = project_root / "runs" / "detected_popup_outputs"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / f"{Path(image_path).stem}_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
cv2.imwrite(str(save_path), img_bgr)
print(f"Saved: {save_path}")

# Display
if not args["no_display"]:
    cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection Result", 1000, 800)
    cv2.imshow("Detection Result", img_bgr)
    if args["auto_close"] > 0:
        cv2.waitKey(args["auto_close"] * 1000)
    else:
        cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Done!")