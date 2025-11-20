import os
import numpy as np
import joblib
import csv
from sklearn.cluster import KMeans
from collections import Counter

# Paths
PROCESSED_DIR = os.path.join("data", "processed")
REDUCED_FILE = os.path.join(PROCESSED_DIR, "reduced_features.npy")
FILENAMES_FILE = os.path.join(PROCESSED_DIR, "filenames.npy")
LABELS_FILE = os.path.join(PROCESSED_DIR, "labels.npy")
CLUSTERS_FILE = os.path.join(PROCESSED_DIR, "cluster_assignments.npy")
CLUSTER_SUMMARY_CSV = os.path.join(PROCESSED_DIR, "cluster_summary.csv")

MODEL_DIR = "models"
KMEANS_MODEL_FILE = os.path.join(MODEL_DIR, "kmeans.pkl")


def _ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)


def perform_clustering(n_clusters=6):
    """
    Perform KMeans clustering on PCA-reduced features, save model and assignments.
    Produces a CSV with: image_path, original_label, cluster_id
    """
    _ensure_dirs()

    if not os.path.exists(REDUCED_FILE):
        raise FileNotFoundError(f"Reduced feature file not found: {REDUCED_FILE}. Run PCA reduction first.")

    print("Loading reduced features...")
    X = np.load(REDUCED_FILE)
    print("Reduced features shape:", X.shape)

    print(f"Clustering with KMeans (n_clusters={n_clusters}) ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X)

    # Save model and assignments
    joblib.dump(kmeans, KMEANS_MODEL_FILE, compress=3)
    np.save(CLUSTERS_FILE, cluster_ids)

    print(f"KMeans model saved -> {KMEANS_MODEL_FILE}")
    print(f"Cluster assignments saved -> {CLUSTERS_FILE}")

    # If filenames/labels exist, create a CSV summary
    filenames = np.load(FILENAMES_FILE, allow_pickle=True) if os.path.exists(FILENAMES_FILE) else None
    labels = np.load(LABELS_FILE, allow_pickle=True) if os.path.exists(LABELS_FILE) else None

    if filenames is not None:
        print("Creating cluster summary CSV...")
        with open(CLUSTER_SUMMARY_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "original_label", "cluster_id"])
            for i, fname in enumerate(filenames):
                orig_label = labels[i] if labels is not None else ""
                writer.writerow([fname, orig_label, int(cluster_ids[i])])
        print(f"Cluster summary CSV -> {CLUSTER_SUMMARY_CSV}")

    # Print cluster counts
    counts = Counter(cluster_ids)
    print("Cluster distribution:")
    for k, v in sorted(counts.items()):
        print(f"  Cluster {k}: {v} images")

    print("Clustering completed.")
