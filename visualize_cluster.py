import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Plotting a scatter plot of two features (you can choose any two features), colored by cluster
sns.scatterplot(data=segmented_features_df, x='chroma_stft', y='spectral_centroid', hue='cluster', palette='viridis')
plt.title('Cluster Representation in Feature Space')
plt.xlabel('Chroma STFT')
plt.ylabel('Spectral Centroid')
plt.legend(title='Cluster')
plt.show()

# Load features DataFrame from the CSV file
features_df = pd.read_csv('/content/drive/MyDrive/audio_features1.csv')

# Extract relevant features for clustering
feature_columns = features_df.columns[1:]  # Assuming first column is 'filename'
X = features_df[feature_columns].values

# Perform GMM clustering
n_components = 5  # Adjust as needed
gmm = GaussianMixture(n_components=n_components, random_state=0).fit(X)
labels = gmm.predict(X)

# Add cluster labels to the DataFrame
features_df['cluster'] = labels

# Save the segmented features with cluster labels to a CSV file
features_df.to_csv('/content/drive/MyDrive/segmented_audio_features1.csv', index=False)

# Evaluate clustering with silhouette score and Davies-Bouldin index
silhouette_avg = silhouette_score(X, labels)
davies_bouldin_avg = davies_bouldin_score(X, labels)
print(f'Silhouette Score: {silhouette_avg}')
print(f'Davies-Bouldin Index: {davies_bouldin_avg}')

# Reduce dimensions using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA results
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.title('GMM Clustering (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Reduce dimensions using t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# Plot t-SNE results
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=50)
plt.title('GMM Clustering (t-SNE)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster Label')
plt.show()
