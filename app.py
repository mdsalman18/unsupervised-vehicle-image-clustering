from src.extract_features import extract_and_save_features
from src.pca_reduce import apply_pca_and_save
from src.cluster import perform_clustering

def main():
    extract_and_save_features()
    apply_pca_and_save(n_components=100)
    perform_clustering(n_clusters=8)

if __name__ == "__main__":
    main()
