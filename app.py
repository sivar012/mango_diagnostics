from flask import Flask, render_template, request
import os

# Import custom utilities
from utils.tree_detect import detect_mango_trees
from utils.ndvi_predict import predict_ndvi
from utils.disease_predict import predict_disease
from utils.pest_predict import predict_pest
from utils.weed_detect import detect_weeds

# Create Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded image
        img = request.files["image"]
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)

        # Run all model predictions
        results = {
            "tree": detect_mango_trees(img_path),
            "ndvi": predict_ndvi(img_path),
            "disease": predict_disease(img_path),
            "pest": predict_pest(img_path),
            "weed": detect_weeds(img_path),
            "image_path": img_path
        }

        return render_template("result.html", results=results)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
