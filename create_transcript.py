# create_transcript.py
import os
from dotenv import load_dotenv
from transformers import pipeline
import librosa
import datetime

# Load environment variables from .env file
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Load the ASR model directly from Hugging Face
asr = pipeline(task="automatic-speech-recognition", model="openai/whisper-large")

# Ensure the sampling rate is correct
sampling_rate = asr.feature_extractor.sampling_rate

# Define the function to transcribe and save audio files
def process_audio_files():
    audio_dir = './data/audio'
    transcription_dir = './data/transcription'

    if not os.path.exists(transcription_dir):
        os.makedirs(transcription_dir)

    processed_files = set()

    # List all audio files in the directory
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav'))]

    for audio_file in audio_files:
        if audio_file in processed_files:
            continue  # Skip already processed files

        filepath = os.path.join(audio_dir, audio_file)

        # Load the audio file
        audio, sr = librosa.load(filepath, sr=sampling_rate)

        # Transcribe the audio
        output = asr(audio, return_timestamps=True)
        transcription = output["text"]

        # Store the transcribed text in a file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        transcription_filename = f'transcription_{timestamp}_{audio_file}.txt'
        transcription_path = os.path.join(transcription_dir, transcription_filename)

        with open(transcription_path, 'w') as f:
            f.write(transcription)  # Write the raw transcription directly

        # Mark the file as processed
        processed_files.add(audio_file)
        print(f"Processed and transcribed: {audio_file}")

# Run the audio processing function
if __name__ == "__main__":
    process_audio_files()