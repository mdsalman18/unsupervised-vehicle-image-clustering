import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths
PROCESSED_DIR = os.path.join("data", "processed")
FEATURES_FILE = os.path.join(PROCESSED_DIR, "features.npy")
REDUCED_FILE = os.path.join(PROCESSED_DIR, "reduced_features.npy")

MODEL_DIR = "models"
PCA_MODEL_FILE = os.path.join(MODEL_DIR, "pca.pkl")
SCALER_MODEL_FILE = os.path.join(MODEL_DIR, "scaler.pkl")


def _ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)


def apply_pca_and_save(n_components=100):
    _ensure_dirs()

    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}. Run extract_features first.")

    print("Loading features...")
    X = np.load(FEATURES_FILE)
    print("Features shape:", X.shape)

    # Safety check for n_components
    if n_components > X.shape[1]:
        n_components = X.shape[1]
        print(f"Warning: n_components > n_features, setting n_components = {X.shape[1]}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("StandardScaler fitted.")

    # Apply PCA
    print(f"Applying PCA with n_components={n_components} ...")
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    print("PCA done. Reduced shape:", X_reduced.shape)

    # Save outputs
    np.save(REDUCED_FILE, X_reduced)
    joblib.dump(pca, PCA_MODEL_FILE, compress=3)
    joblib.dump(scaler, SCALER_MODEL_FILE, compress=3)

    print(f"Saved reduced features -> {REDUCED_FILE}")
    print(f"Saved PCA model         -> {PCA_MODEL_FILE}")
    print(f"Saved Scaler model      -> {SCALER_MODEL_FILE}")
