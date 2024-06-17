# Import necessary libraries
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import os
import soundfile as sf
from spleeter.separator import Separator

# Function to perform segmentation using Gaussian Mixture Models (GMM)
def segment_audio_features_gmm(features_df, n_components=5):
    # Extract relevant features for clustering
    feature_columns = features_df.columns[1:]
    X = features_df[feature_columns].values

    # Perform GMM clustering
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    labels = gmm.predict(X)

    # Add cluster labels to the DataFrame
    features_df['cluster'] = labels

    return features_df

# Load features DataFrame from the previous step
features_df = pd.read_csv('/content/drive/MyDrive/audio_features1.csv')

# Segment the audio features
segmented_features_df = segment_audio_features_gmm(features_df, n_components=5)

# Save the segmented features to a CSV file
segmented_features_df.to_csv('/content/drive/MyDrive/segmented_audio_features1.csv', index=False)

# Initialize Spleeter separator
separator = Separator('spleeter:2stems')

# Directory containing your audio files
audio_dir = '/content/drive/MyDrive/archive (5)/genrenew'

# List all audio files
audio_files = []
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith('.wav') or file.endswith('.mp3'):
            audio_files.append(os.path.join(root, file))

# Extract background music from the audio files and save to disk
output_dir = '/content/drive/MyDrive/background_music1'
os.makedirs(output_dir, exist_ok=True)

for file in audio_files:
    separator.separate_to_file(file, output_dir)

print("Background music extraction complete. Saved to background_music1 directory")
