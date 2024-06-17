# Import necessary libraries
import os
import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal

# Define the directory containing the extracted background music files
output_dir = '/content/drive/MyDrive/background_music1'

def post_process_audio(audio, sr=22050):
    # Apply a low-pass filter to remove high-frequency noise
    sos = signal.butter(10, 0.1, 'low', output='sos')
    filtered_audio = signal.sosfilt(sos, audio)

    return filtered_audio

# Post-process extracted background music
for file in os.listdir(output_dir):
    if file.endswith('.wav'):
        file_path = os.path.join(output_dir, file)
        audio, sr = librosa.load(file_path, sr=22050)
        refined_audio = post_process_audio(audio, sr=sr)
        sf.write(file_path, refined_audio, sr)

print("Post-processing complete. Refined audio saved to background_music directory.")
