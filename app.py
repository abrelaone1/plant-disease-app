import os
import json
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model and labels
model = load_model("plant_disease_model.keras")
with open("plant_disease_label_map.json") as f:
    label_map = json.load(f)["classes"]

IMG_SIZE = (300, 300)

# Disease suggestions (Ethiopia-friendly)
disease_info = {
    "Tomato___Late_blight": {"explanation":"Fungal disease causing dark lesions.", "suggestion":"Remove infected leaves, apply copper fungicide."},
    "Potato___Early_blight": {"explanation":"Alternaria fungus causing brown spots.", "suggestion":"Crop rotation, remove affected leaves, apply Mancozeb."},
    "Potato___Late_blight": {"explanation":"Phytophthora blight, spreads fast.", "suggestion":"Destroy infected plants, apply Metalaxyl."},
    "Corn___Northern_Leaf_Blight": {"explanation":"Gray lesions reduce yield.", "suggestion":"Rotate crops, plant resistant varieties."},
    "Corn___Common_rust": {"explanation":"Reddish-brown pustules.", "suggestion":"Plant resistant varieties, apply fungicide if severe."},
    "Tomato___Healthy": {"explanation":"Healthy tomato plant.", "suggestion":"Regular irrigation, monitor pests."},
    "Potato___Healthy": {"explanation":"Healthy potato plant.", "suggestion":"Maintain field hygiene, monitor regularly."},
    "Corn___Healthy": {"explanation":"Healthy maize plant.", "suggestion":"Balanced fertilizer, weed regularly."}
}

# Helper functions
def get_crop_classes(label):
    crop = label.split("___")[0]
    return [cls for cls in label_map if cls.startswith(crop)]

def predict_disease(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    x = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    label = label_map[idx]
    confidence = float(preds[idx])
    crop_classes = get_crop_classes(label)
    probs = {cls: float(preds[label_map.index(cls)]) for cls in crop_classes}
    return label, confidence, probs

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files: return "No file"
        file = request.files["file"]
        if file.filename == "": return "No selected file"
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        label, confidence, probs = predict_disease(filepath)
        info = disease_info.get(label, {"explanation":"No info available", "suggestion":"No suggestion available"})
        return render_template("result.html",
                               image_url=filepath,
                               label=label,
                               confidence=round(confidence*100,2),
                               explanation=info["explanation"],
                               suggestion=info["suggestion"],
                               probs=probs)
    return render_template("index.html")

if __name__ == "__main__":
    app.run()