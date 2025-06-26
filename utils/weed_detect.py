from ultralytics import YOLO

# ✅ Correct model path with raw string for Windows path
model = YOLO(r"C:\Users\Admin\OneDrive\Desktop\hackthon\mango_diagnostics_app\models\yolo12m.pt")

def detect_weeds(img_path):
    results = model.predict(img_path, conf=0.3)
    count = len(results[0].boxes)
    return f"{count} weed(s) detected"
