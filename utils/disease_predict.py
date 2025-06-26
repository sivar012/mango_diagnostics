from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

# ✅ Load your trained model
model = load_model(r"C:\Users\Admin\OneDrive\Desktop\hackthon\mango_diagnostics_app\models\efficientnetb0-Mango_Diseases-100.00.keras")

# ✅ Class names mapping
class_names = {
    0: 'Anthracnose',
    1: 'Bacterial Canker',
    2: 'Cutting Weevil',
    3: 'Die Back',
    4: 'Gall Midge',
    5: 'Healthy',
    6: 'Powdery Mildew',
    7: 'Sooty Mould'
}

# ✅ Main prediction function
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))
    pred = model.predict(img_array)
    predicted_index = np.argmax(pred)
    return f"Disease Prediction: {class_names[predicted_index]}"
