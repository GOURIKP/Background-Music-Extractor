# Import necessary libraries
import os
import librosa
import numpy as np
import pandas as pd
import noisereduce as nr
import matplotlib.pyplot as plt
import librosa.display
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the directory containing your audio files
audio_dir = r'/content/drive/MyDrive/archive (5)/genrenew'

# List all audio files
audio_files = []
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith('.wav') or file.endswith('.mp3'):
            audio_files.append(os.path.join(root, file))

# Function to preprocess and extract features from an audio file
def extract_features(file_path, target_sr=22050):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=target_sr)  # Resample during load

        # Normalize audio
        y = librosa.util.normalize(y)

        # Reduce noise
        y = nr.reduce_noise(y=y, sr=sr)

        # Extract STFT spectrogram
        stft = np.abs(librosa.stft(y))
        stft_db = librosa.amplitude_to_db(stft, ref=np.max)

        # Extract Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # Compute mean of MFCCs
        mfccs_mean = np.mean(mfccs, axis=1)

        # Store features
        features = {
            'filename': os.path.basename(file_path),
            'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            'rmse': np.mean(librosa.feature.rms(y=y)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
        }

        # Add MFCCs to the feature dictionary
        for i in range(1, 21):
            features[f'mfcc{i}'] = mfccs_mean[i-1]

        return features, stft_db, mel_db

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

# Function to process files in parallel
def process_files_in_parallel(files, max_workers=4):
    features_list = []
    spectrograms_list = []
    mel_spectrograms_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(extract_features, file): file for file in files}

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                features, stft_db, mel_db = future.result()
                if features is not None:
                    features_list.append(features)
                    spectrograms_list.append((features['filename'], stft_db))
                    mel_spectrograms_list.append((features['filename'], mel_db))
            except Exception as exc:
                print(f"{file} generated an exception: {exc}")

    return features_list, spectrograms_list, mel_spectrograms_list

# Extract features from all audio files in parallel
features_list, spectrograms_list, mel_spectrograms_list = process_files_in_parallel(audio_files, max_workers=4)

# Convert feature list to DataFrame
features_df = pd.DataFrame(features_list)

# Save features to a CSV file
features_df.to_csv('/content/drive/MyDrive/audio_features.csv1', index=False)

print("Feature extraction complete. Saved to audio_features.csv")

# Optional: Save spectrogram and mel spectrogram images
output_dir = '/content/drive/MyDrive/spectrograms1'
os.makedirs(output_dir, exist_ok=True)

for filename, stft_db in spectrograms_list:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(stft_db, sr=22050, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'STFT Spectrogram: {filename}')
    plt.savefig(os.path.join(output_dir, f'{filename}_stft.png'))
    plt.close()

output_dir = '/content/drive/MyDrive/mel_spectrograms1'
os.makedirs(output_dir, exist_ok=True)

for filename, mel_db in mel_spectrograms_list:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=22050, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram: {filename}')
    plt.savefig(os.path.join(output_dir, f'{filename}_mel.png'))
    plt.close()
