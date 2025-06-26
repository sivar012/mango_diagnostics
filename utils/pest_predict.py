from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

# ✅ Correct model path
MODEL_PATH = r"C:\Users\Admin\OneDrive\Desktop\hackthon\mango_diagnostics_app\models\pest_mobilenetv2_model.h5"
model = load_model(MODEL_PATH)

# ✅ Class labels
class_labels = ['aphids', 'armyworm', 'beetle', 'bollworm',
                'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem borer']

def predict_pest(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "Image not found"

    # ✅ Resize to match model input: 224x224
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

    # ✅ Predict
    prediction = model.predict(img)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index] * 100

    return f"{class_labels[predicted_index]} ({confidence:.2f}%)"
