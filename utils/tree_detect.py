from ultralytics import YOLO

# Use raw string or forward slashes
MODEL_PATH = r"C:\Users\Admin\OneDrive\Desktop\hackthon\mango_diagnostics_app\models\best.pt"

model = YOLO(MODEL_PATH)

def detect_mango_trees(img_path):
    results = model.predict(img_path, conf=0.3)
    count = len(results[0].boxes)
    return f"{count} tree(s) detected"
