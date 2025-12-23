import numpy as np
import os
from src.engine import CustomPCA
from src.loader import load_spotify_data
from src.visualization import plot_projection, plot_correlation_circle

def main():
    if not os.path.exists('assets'):
        os.makedirs('assets')

    # 1. load data
    print("--- loading data ---")
    fpath = "data/data_spotify.csv"
    
    if not os.path.exists(fpath):
        print(f"error: {fpath} not found.")
        return

    X, genres, feats, df = load_spotify_data(fpath)
    print(f"loaded {X.shape[0]} tracks, {X.shape[1]} features")

    # 2. train pca
    print("\n--- training pca ---")
    pca = CustomPCA(n_components=2)
    pca.fit(X, feats)
    print(f"variance ratio: {pca.explained_variance_ratio[:2]}")

    # 3. transform & viz
    X_proj = pca.transform(X)
    
    print("\n--- plotting ---")
    plot_projection(X_proj, genres, "Latent Space Projection", "assets/pca_projection.png")
    
    corr = pca.get_components_stats()
    plot_correlation_circle(corr, feats, "assets/correlation_circle.png")

    # 4. inference test
    print("\n--- inference test ---")
    new_data = np.array([
        [0.665, 0.91, 6, -4.682, 0.223, 0.0394, 0.0, 0.227, 0.887, 84.099, 293973, 4],
        [0.575, 0.286, 9, -6.768, 0.0322, 0.856, 0.0, 0.179, 0.591, 141.574, 183477, 4],
        [0.313, 0.00661, 4, -39.292, 0.0524, 0.994, 0.933, 0.102, 0.318, 110.931, 173137, 4]
    ])
    
    new_proj = pca.transform(new_data)
    print("projected coords:")
    print(new_proj)

    print("\nsuccess.")

if __name__ == "__main__":
    main()