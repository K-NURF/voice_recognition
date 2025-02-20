from pyannote.audio import Pipeline
from pydub import AudioSegment

# Load the pretrained diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="")# Path to your audio file
audio_file_path = "/home/k_nurf/voice_recognition/media/1737297024.186904.wav"

# Perform speaker diarization
diarization = pipeline(audio_file_path)

# Print the diarization results
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker} speaks from {turn.start:.2f}s to {turn.end:.2f}s")

# Load the audio file
audio = AudioSegment.from_wav(audio_file_path)

# Create directory to save the chunks
import os
output_dir = "diarized_segments"
os.makedirs(output_dir, exist_ok=True)

# Export each segment
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_time = turn.start * 1000  # Convert seconds to milliseconds
    end_time = turn.end * 1000      # Convert seconds to milliseconds
    
    # Extract the audio chunk
    chunk = audio[start_time:end_time]
    
    # Save the chunk with speaker label
    chunk.export(f"{output_dir}/speaker_{speaker}_{int(turn.start)}-{int(turn.end)}.wav", format="wav")
    print(f"Exported: speaker_{speaker}_{int(turn.start)}-{int(turn.end)}.wav")
