import os
import numpy as np
from tqdm import tqdm
from PIL import Image

# Keras (with JAX backend)
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import img_to_array
from keras.models import Model

# Paths (adjust if needed)
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
FEATURES_FILE = os.path.join(PROCESSED_DIR, "features.npy")
LABELS_FILE = os.path.join(PROCESSED_DIR, "labels.npy")
FILENAMES_FILE = os.path.join(PROCESSED_DIR, "filenames.npy")


def _ensure_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_vgg16_feature_model():
    """Load VGG16 (ImageNet weights) without top classification head."""
    base = VGG16(weights="imagenet", include_top=False)
    model = Model(inputs=base.input, outputs=base.output)
    return model


def _preprocess_image_to_model(path, target_size=(224, 224)):
    """Open image, convert to RGB, resize, convert to array and preprocess."""
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def extract_and_save_features():
    """
    Walk data/raw/, extract features for every image using VGG16 (no top),
    and save features, labels, and filenames into data/processed/.
    """
    _ensure_dirs()
    model = load_vgg16_feature_model()
    print("Loaded VGG16 feature extractor (Keras + JAX backend).")

    features = []
    labels = []
    filenames = []

    if not os.path.isdir(RAW_DATA_DIR):
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    classes = sorted([
        d for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ])

    if not classes:
        raise RuntimeError(f"No class folders found under {RAW_DATA_DIR}")

    print("Classes found:", classes)

    total_images = sum(
        len([f for f in os.listdir(os.path.join(RAW_DATA_DIR, cls))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for cls in classes
    )

    print(f"Processing ~{total_images} images... (CPU/JAX backend)")

    for cls in classes:
        class_folder = os.path.join(RAW_DATA_DIR, cls)
        for fname in tqdm(sorted(os.listdir(class_folder)), desc=f"Extracting [{cls}]"):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(class_folder, fname)
            try:
                img_array = _preprocess_image_to_model(path)
                feat = model.predict(img_array, verbose=0)
                feat = feat.flatten()

                features.append(feat)
                labels.append(cls)
                filenames.append(path)

            except Exception as e:
                print(f"Skipped {path} due to error: {e}")

    features = np.array(features)
    labels = np.array(labels)
    filenames = np.array(filenames)

    np.save(FEATURES_FILE, features)
    np.save(LABELS_FILE, labels)
    np.save(FILENAMES_FILE, filenames)

    print("\nFeature extraction finished.")
    print(f"Saved features -> {FEATURES_FILE} (shape: {features.shape})")
    print(f"Saved labels   -> {LABELS_FILE} (n: {labels.shape[0]})")
    print(f"Saved filenames-> {FILENAMES_FILE} (n: {filenames.shape[0]})")
