# Function to extract background music from a single audio file
def extract_background_music(audio_file_path, output_dir):
    # Initialize Spleeter separator
    separator = Separator('spleeter:2stems')

    # Extract background music from the audio file and save to disk
    os.makedirs(output_dir, exist_ok=True)
    separator.separate_to_file(audio_file_path, output_dir)

    print("Background music extraction complete. Saved to background_music directory")

# Path to the new audio file
audio_file_path = 'Add any audio path file'
# Output directory for the extracted background music
output_dir = '/content/drive/MyDrive/background_music1'

# Extract background music from the new audio file
extract_background_music(audio_file_path, output_dir)
