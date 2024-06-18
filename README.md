# Background-Music-Extractor
An application to extract background audio from music files 

This project involves building a background music extractor for audio files. The process includes extracting audio features, performing Gaussian Mixture Model (GMM) clustering, separating background music using Spleeter, and post-processing the extracted background music. Additionally, the project includes visualizing the clustered audio features using PCA and t-SNE.

## Dataset

The dataset used in this project is the [Indian Music Genre Dataset](https://www.kaggle.com/datasets/winchester19/indian-music-genre-dataset) from Kaggle. Download the dataset and place the audio files in a directory for processing.

## Prerequisites

- Python 3.7+
- Libraries: `librosa`, `numpy`, `pandas`, `noisereduce`, `matplotlib`, `seaborn`, `scikit-learn`, `soundfile`, `spleeter`, `scipy`

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/background-music-extractor.git
    cd background-music-extractor
    ```

2. Install the required libraries:

    ```bash
    pip install librosa numpy pandas noisereduce matplotlib seaborn scikit-learn soundfile spleeter scipy
    ```

3. Install FFmpeg (required for Spleeter):

    ```bash
    sudo apt-get install ffmpeg
    ```

## Usage

### Extracting Audio Features

1. Define the directory containing your audio files in the script:

    ```python
    audio_dir = r'/path/to/your/audio/files'
    ```

2. Run the script to extract audio features:

    ```bash
    python extract_features.py
    ```

3. The extracted features will be saved to `audio_features.csv`.

### Performing GMM Clustering and Extracting Background Music

1. Run the script to perform GMM clustering on the extracted features:

    ```bash
    python gmm_and_extract.py
    ```

2. The segmented features with cluster labels will be saved to `segmented_audio_features.csv`.
3. The extracted background music files will be saved to `background_music` directory.

### Post-processing Background Music

1. Run the script to post-process the extracted background music:

    ```bash
    python post_process_audio.py
    ```

2. The post-processed audio files will be saved to `background_music` directory.

### Visualizing Clusters

1. Run the script to visualize the clusters using PCA and t-SNE:

    ```bash
    python visualize_clusters.py
    ```

2. The scatter plots of the clusters will be displayed.

## Files

- `extract_features.py`: Script to extract audio features.
- `gmm_and_extract.py`: Script to perform GMM clustering and extract background music using Spleeter.
- `post_process_audio.py`: Script to post-process the extracted background music.
- `visualize_clusters.py`: Script to visualize the clusters using PCA and t-SNE.
- `audio_features.csv`: Extracted audio features.
- `segmented_audio_features.csv`: Segmented audio features with cluster labels.
