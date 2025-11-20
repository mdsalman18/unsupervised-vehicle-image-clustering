import os

from src.extract_features import extract_and_save_features
from src.pca_reduce import apply_pca_and_save
from src.cluster import perform_clustering
from src.predict_new import predict_new_image

FEATURES_FILE = "data/processed/features.npy"
REDUCED_FILE = "data/processed/reduced_features.npy"
CLUSTERS_FILE = "data/processed/cluster_assignments.npy"

def main():

    print("\n📌 STEP 1: Feature Extraction")
    if not os.path.exists(FEATURES_FILE):
        extract_and_save_features()
    else:
        print("✔ Features already extracted — skipping.")

    print("\n📌 STEP 2: PCA Reduction")
    if not os.path.exists(REDUCED_FILE):
        apply_pca_and_save(n_components=100)
    else:
        print("✔ PCA reduced features already exist — skipping.")

    print("\n📌 STEP 3: Clustering")
    if not os.path.exists(CLUSTERS_FILE):
        perform_clustering(n_clusters=8)
    else:
        print("✔ Clusters already exist — skipping.")

    print("\n📌 STEP 4: Prediction on New Image")

    # 🔥 CHANGE THIS PATH for prediction
    test_image_path = "new.jpg"

    if os.path.exists(test_image_path):
        predict_new_image(test_image_path)
    else:
        print(f"⚠ Test image not found at: {test_image_path}")

    print("\n🎉 Pipeline + Prediction Completed!")


if __name__ == "__main__":
    main()
