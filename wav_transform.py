import os
from pydub import AudioSegment


def convert_to_wav(input_folder, output_folder, sample_rate=22050):
    """
    Convert all .opus files in a folder to .wav format.

    Args:
        input_folder (str): Path to folder containing .opus files.
        output_folder (str): Path to save .wav files.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        opus_path = os.path.join(input_folder, filename)
        wav_filename = os.path.splitext(filename)[0] + ".wav"
        wav_path = os.path.join(output_folder, wav_filename)

        # Convert using pydub (requires ffmpeg)
        try:
            audio = AudioSegment.from_file(opus_path, codec="opus")
            audio = audio.set_frame_rate(sample_rate)
            audio.export(wav_path, format="wav")
            print(f"Converted: {filename} → {wav_filename}")
        except Exception as e:
            print(f"Failed to convert {filename}: {str(e)}")


# Example usage
input_folder = r"C:\Users\hp\Desktop\JAMILA" # Replace with your folder
output_folder = r"C:\Users\hp\Desktop\jamila_wav" # Replace with output folder
convert_to_wav(input_folder, output_folder)