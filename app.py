from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import pickle
import joblib
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow_hub as hub
from sklearn.neighbors import NearestNeighbors
from werkzeug.utils import secure_filename

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Create upload folder if not exists
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load style data and models
styles_df = pd.read_csv("styles.csv", usecols=["id", "articleType"])
styles_dict = dict(zip(styles_df["id"].astype(str), styles_df["articleType"]))

with open("features_cache_pca.pkl", "rb") as f:
    data = pickle.load(f)
    image_paths = data["image_paths"]
    features = data["features"]

pca = joblib.load("pca_model.pkl")

# Load TensorFlow Hub EfficientNet
efficientnet_extractor = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1")

# Fit KNN
knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(features)

def extract_feature(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = efficientnet_extractor(img_array).numpy()
    return pca.transform(features)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    if not img_file:
        return "No file uploaded", 400

    filename = secure_filename(img_file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    img_file.save(upload_path)

    query_feature = extract_feature(upload_path)
    distances, indices = knn.kneighbors(query_feature)

    recommended_paths = [image_paths[i] for i in indices[0]]
    recommended_filenames = [os.path.basename(p) for p in recommended_paths]
    predicted_styles = [styles_dict.get(f.split(".")[0], "Unknown") for f in recommended_filenames]

    results = [
        {"path": f"images/{fname}", "label": style, "distance": round(float(dist), 4)}
        for fname, style, dist in zip(recommended_filenames, predicted_styles, distances[0])
    ]

    return render_template("results.html", uploaded_img=f"uploads/{filename}", results=results)

if __name__ == '__main__':
    app.run(debug=True)

