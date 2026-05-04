import os
import pathlib 
from scipy.io.wavfile import write
from src.preprocessing import Preprocess

base_dir = pathlib.Path.home() / "VoicePathologyClassification" / "VoicePathologyClassification" / "data"
raw_dir = base_dir / "raw"
processed_dir = base_dir / "processed"

for root, dirs, files in os.walk(raw_dir):
    for file in files:
        if not file.lower().endswith(".wav"):
            print("The file " + file + "is not a .wav file and can not be processed.")
            continue
        audio = Preprocess(os.path.join(root, file),0.1,60)
        processed_audio, metadata = audio.process()
        
        out_dir = os.path.join(processed_dir, file)
        print(out_dir)
        write(out_dir, 16000, processed_audio)
    print("Directory preprocess complete.")



