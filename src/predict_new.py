import os
import numpy as np
from PIL import Image
import joblib

from keras.applications.vgg16 import preprocess_input, VGG16
from keras.utils import img_to_array
from keras.models import Model

# Paths
MODEL_DIR = "models"
SCALER_MODEL_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
PCA_MODEL_FILE = os.path.join(MODEL_DIR, "pca.pkl")
KMEANS_MODEL_FILE = os.path.join(MODEL_DIR, "kmeans.pkl")


def load_models():
    base = VGG16(weights="imagenet", include_top=False)
    feature_model = Model(inputs=base.input, outputs=base.output)

    scaler = joblib.load(SCALER_MODEL_FILE)
    pca = joblib.load(PCA_MODEL_FILE)
    kmeans = joblib.load(KMEANS_MODEL_FILE)

    return feature_model, scaler, pca, kmeans


def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predict_new_image(image_path):
    feature_model, scaler, pca, kmeans = load_models()

    # Step 1 → Preprocess image
    img_arr = preprocess_image(image_path)

    # Step 2 → Extract deep features
    features = feature_model.predict(img_arr, verbose=0).flatten().reshape(1, -1)

    # Step 3 → Scale
    features_scaled = scaler.transform(features)

    # Step 4 → PCA reduction
    features_pca = pca.transform(features_scaled)

    # Step 5 → KMeans prediction
    cluster_id = kmeans.predict(features_pca)[0]

    print(f"Predicted Cluster for {image_path}: {cluster_id}")
    return cluster_id
